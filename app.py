import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fitparse import FitFile

st.set_page_config(page_title="Aero Analyzer (Chung Method)", layout="wide")

# --- Hilfsfunktionen ---
@st.cache_data
def parse_fit_file(uploaded_file):
    fitfile = FitFile(uploaded_file)
    records = []
    laps_timestamps = []
    
    # 1. Runden auslesen
    for lap in fitfile.get_messages('lap'):
        lap_data = {}
        for data_field in lap:
            lap_data[data_field.name] = data_field.value
        if 'timestamp' in lap_data:
            laps_timestamps.append(lap_data['timestamp'])

    # 2. Kontinuierliche Daten auslesen
    for record in fitfile.get_messages('record'):
        data = {}
        for data_field in record:
            data[data_field.name] = data_field.value
        records.append(data)
        
    df = pd.DataFrame(records)
    
    # Notwendige Spalten standardisieren
    if 'speed' in df.columns: df['speed_ms'] = df['speed']
    if 'distance' in df.columns: df['distance_m'] = df['distance']
    if 'altitude' in df.columns: df['altitude_m'] = df['altitude']
    if 'power' in df.columns: df['power_w'] = df['power']
    
    # GPS Daten extrahieren (FIT speichert Semicircles)
    if 'position_lat' in df.columns and 'position_long' in df.columns:
        df['lat'] = df['position_lat'] * (180.0 / (2**31))
        df['lon'] = df['position_long'] * (180.0 / (2**31))
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Runden-Distanzen für den Plot ermitteln
    lap_distances = []
    for lt in laps_timestamps:
        lap_row = df[df['timestamp'] <= pd.to_datetime(lt)]
        if not lap_row.empty and 'distance_m' in lap_row.columns:
            lap_distances.append(lap_row.iloc[-1]['distance_m'])
            
    # Delta-t berechnen
    df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(1.0)
    df.loc[df['dt'] <= 0, 'dt'] = 1.0
    
    df = df.dropna(subset=['speed_ms', 'power_w', 'distance_m']).reset_index(drop=True)
    return df, lap_distances

def calculate_wind_and_ve(df, cda, crr, m, rho, eta, v_wind_kmh, wind_dir_deg):
    v = df['speed_ms'].values
    p = df['power_w'].values
    dt = df['dt'].values
    
    # Relative Luftgeschwindigkeit (v_air) berechnen
    v_wind_ms = v_wind_kmh / 3.6
    
    if 'lat' in df.columns and 'lon' in df.columns and v_wind_ms > 0:
        lat_rad = np.radians(df['lat'])
        lon_rad = np.radians(df['lon'])
        dlon = lon_rad.diff()
        lat1 = lat_rad.shift(1)
        lat2 = lat_rad
        
        # Kompasskurs (Heading) berechnen
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
        bearing = bearing.bfill()
        
        # Gegenwindkomponente = Windgeschw. * cos(Fahrtrichtung - Windherkunft)
        headwind_comp = v_wind_ms * np.cos(np.radians(bearing - wind_dir_deg))
        v_air = v + headwind_comp
        # Verhindern, dass bei extremem Rückenwind v_air negativ wird (Aerodynamik dreht sich nicht um)
        v_air = np.maximum(v_air, 0.0)
    else:
        v_air = v

    m_eff = m * 1.03 # Rotationsmasse
    
    p_wheel = p * eta
    p_rr = m * 9.81 * crr * v
    
    # Leistung Luftwiderstand: P_aero = 0.5 * rho * CdA * (v_air^2) * v_ground
    p_aero = 0.5 * rho * cda * (v_air**2) * v
    
    dv = np.diff(v, prepend=v[0])
    p_acc = m_eff * v * (dv / dt)
    
    p_climb = p_wheel - p_rr - p_aero - p_acc
    
    dh = (p_climb / (m * 9.81)) * dt
    dh[v < 1.0] = 0.0
    
    df['v_elev'] = np.cumsum(dh)
    if 'altitude_m' in df.columns:
        df['v_elev'] += df['altitude_m'].iloc[0]
        
    return df

# --- UI Aufbau ---
st.title("🚴‍♂️ Aero Analyzer (Chung VE Method)")

with st.sidebar:
    st.header("1. Rider & Bike")
    m_fahrer = st.number_input("Gewicht Fahrer (kg)", value=85.0)
    m_rad = st.number_input("Gewicht Rad + Ausrüstung (kg)", value=11.0)
    
    st.header("2. Environment")
    crr = st.number_input("Rollwiderstand (Crr)", value=0.0039, format="%.5f", step=0.0001)
    rho = st.number_input("Luftdichte (kg/m³)", value=1.162, format="%.3f")
    eta = st.number_input("Antriebseffizienz", value=0.975, format="%.3f")
    
    st.header("3. Wind (Optional)")
    v_wind_kmh = st.number_input("Windgeschwindigkeit (km/h)", value=0.0, step=1.0)
    wind_dir_deg = st.number_input("Windrichtung (Herkunft in Grad, N=0, O=90, S=180, W=270)", value=158.0, step=1.0)
    st.caption("Beispiel: SSO-Wind (Süd-Süd-Ost) entspricht ca. 158°.")

    st.header("4. CdA Tuning")
    cda_slider = st.slider("CdA Wert (m²)", min_value=0.20, max_value=0.50, value=0.356, step=0.001)

m_system = m_fahrer + m_rad

uploaded_file = st.file_uploader("FIT-Datei hochladen", type=["fit"])

if uploaded_file is not None:
    with st.spinner("Datei wird geparst..."):
        df, lap_distances = parse_fit_file(uploaded_file)
    
    if not 'lat' in df.columns:
        st.warning("Achtung: Keine GPS-Daten in der Datei gefunden. Windkorrektur kann nicht dynamisch berechnet werden.")
    
    st.markdown("### Streckenabschnitt auswählen")
    min_dist, max_dist = float(df['distance_m'].min()), float(df['distance_m'].max())
    range_dist = st.slider("Distanz-Bereich (m)", min_value=min_dist, max_value=max_dist, value=(min_dist, max_dist))
    
    df_filtered = df[(df['distance_m'] >= range_dist[0]) & (df['distance_m'] <= range_dist[1])].copy()
    
    if len(df_filtered) > 0:
        df_calc = calculate_wind_and_ve(df_filtered, cda_slider, crr, m_system, rho, eta, v_wind_kmh, wind_dir_deg)
        
        fig = go.Figure()
        
        # Originales Höhenprofil
        if 'altitude_m' in df_calc.columns:
            fig.add_trace(go.Scatter(x=df_calc['distance_m'], y=df_calc['altitude_m'], 
                                     mode='lines', name='Barometrische Höhe (GPS)',
                                     line=dict(color='gray', width=2)))
            
        # Berechnete Virtual Elevation
        fig.add_trace(go.Scatter(x=df_calc['distance_m'], y=df_calc['v_elev'], 
                                 mode='lines', name='Virtual Elevation (Berechnet)',
                                 line=dict(color='blue', width=3)))
        
        # Runden-Markierungen (Laps)
        for i, lap_dist in enumerate(lap_distances):
            if range_dist[0] <= lap_dist <= range_dist[1]:
                fig.add_vline(x=lap_dist, line_width=1.5, line_dash="dash", line_color="green", 
                              annotation_text=f"Lap {i+1}", annotation_position="top right")
        
        fig.update_layout(title="Höhenprofil vs. Virtual Elevation (Chung)",
                          xaxis_title="Distanz (m)",
                          yaxis_title="Höhe (m)",
                          hovermode="x unified")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Chung-Test Auswertung:** Verstelle den CdA-Slider im Menü, bis die blaue Linie am Anfang und Ende deiner Out-and-Back-Laps (grüne Markierungen) **auf demselben Niveau** liegt. Ignoriere kleine Huckel dazwischen – wichtig ist nur, dass sich die Schleife Start/Ende schließt.")
