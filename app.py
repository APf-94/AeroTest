import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fitparse import FitFile

st.set_page_config(page_title="Aero Test (Chung Method)", layout="wide")

# --- Hilfsfunktionen ---
@st.cache_data
def parse_fit_file(uploaded_file):
    fitfile = FitFile(uploaded_file)
    records = []
    
    for record in fitfile.get_messages('record'):
        data = {}
        for data_field in record:
            data[data_field.name] = data_field.value
        records.append(data)
        
    df = pd.DataFrame(records)
    
    # Notwendige Spalten berechnen
    if 'speed' in df.columns:
        df['speed_ms'] = df['speed']
    if 'distance' in df.columns:
        df['distance_m'] = df['distance']
    if 'altitude' in df.columns:
        df['altitude_m'] = df['altitude']
    if 'power' in df.columns:
        df['power_w'] = df['power']
        
    # Zeitstempel und Delta-t berechnen
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(1.0)
    df.loc[df['dt'] <= 0, 'dt'] = 1.0
    
    # Bereinigen
    df = df.dropna(subset=['speed_ms', 'power_w', 'distance_m']).reset_index(drop=True)
    return df

def calculate_virtual_elevation(df, cda, crr, m, rho, eta):
    v = df['speed_ms'].values
    p = df['power_w'].values
    dt = df['dt'].values
    
    m_eff = m * 1.03 # Berücksichtigung der Rotationsmasse
    
    # Leistungskomponenten
    p_wheel = p * eta
    p_rr = m * 9.81 * crr * v
    p_aero = 0.5 * rho * cda * (v**3)
    
    dv = np.diff(v, prepend=v[0])
    p_acc = m_eff * v * (dv / dt)
    
    # Verbleibende Leistung geht in Steigung (p_climb = m * g * dh/dt)
    p_climb = p_wheel - p_rr - p_aero - p_acc
    
    # Höhenänderung
    dh = (p_climb / (m * 9.81)) * dt
    # Bei v=0 entstehen oft Rechenfehler, daher dh auf 0 setzen
    dh[v < 1.0] = 0.0
    
    df['v_elev'] = np.cumsum(dh)
    # Startpunkt auf gleiche Höhe setzen für den Plot
    if 'altitude_m' in df.columns:
        df['v_elev'] += df['altitude_m'].iloc[0]
        
    return df

# --- UI Aufbau ---
st.title("🚴‍♂️ Aero Analyzer (Chung VE Method)")

with st.sidebar:
    st.header("Parameter")
    m_fahrer = st.number_input("Gewicht Fahrer (kg)", value=85.0)
    m_rad = st.number_input("Gewicht Rad + Ausrüstung (kg)", value=11.0)
    crr = st.number_input("Rollwiderstand (Crr)", value=0.0039, format="%.5f", step=0.0001)
    rho = st.number_input("Luftdichte (kg/m³)", value=1.162, format="%.3f")
    eta = st.number_input("Antriebseffizienz", value=0.975, format="%.3f")
    cda_slider = st.slider("CdA Wert (m²)", min_value=0.20, max_value=0.50, value=0.356, step=0.001)

m_system = m_fahrer + m_rad

uploaded_file = st.file_uploader("FIT-Datei hochladen", type=["fit"])

if uploaded_file is not None:
    with st.spinner("Datei wird geparst..."):
        df = parse_fit_file(uploaded_file)
    
    st.success(f"Datei erfolgreich geladen! {len(df)} Datenpunkte gefunden.")
    
    # Zeitbereichs-Filter, um Test-Laps einzugrenzen
    st.markdown("### Streckenabschnitt auswählen")
    min_dist, max_dist = float(df['distance_m'].min()), float(df['distance_m'].max())
    range_dist = st.slider("Distanz-Bereich (m)", min_value=min_dist, max_value=max_dist, value=(min_dist, max_dist))
    
    df_filtered = df[(df['distance_m'] >= range_dist[0]) & (df['distance_m'] <= range_dist[1])].copy()
    
    if len(df_filtered) > 0:
        # Virtual Elevation berechnen
        df_calc = calculate_virtual_elevation(df_filtered, cda_slider, crr, m_system, rho, eta)
        
        # Plotly Diagramm
        fig = go.Figure()
        
        if 'altitude_m' in df_calc.columns:
            fig.add_trace(go.Scatter(x=df_calc['distance_m'], y=df_calc['altitude_m'], 
                                     mode='lines', name='Barometrische Höhe (GPS)',
                                     line=dict(color='gray', width=2)))
            
        fig.add_trace(go.Scatter(x=df_calc['distance_m'], y=df_calc['v_elev'], 
                                 mode='lines', name='Virtual Elevation (Berechnet)',
                                 line=dict(color='blue', width=3)))
        
        fig.update_layout(title="Höhenprofil vs. Virtual Elevation",
                          xaxis_title="Distanz (m)",
                          yaxis_title="Höhe (m)",
                          hovermode="x unified")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Aktueller CdA:** `{cda_slider:.3f} m²` | **Systemgewicht:** `{m_system:.1f} kg`")
        st.info("💡 **Tipp zur Analyse (Out-and-Back):** Verstelle den CdA-Slider im Seitenmenü so lange, bis die blaue Linie (Virtual Elevation) beim Start- und Wendepunkt auf derselben Höhe ankommt bzw. parallel zum echten barometrischen Höhenprofil verläuft.")
