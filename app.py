import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fitparse import FitFile

st.set_page_config(page_title="Aero Analyzer (Chung Method)", layout="wide")

# --- Initialize Session State for Export ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        'Setup / Run Name', 'CdA', 'Crr', 'Weight (kg)', 
        'Air Density', 'Efficiency', 'Wind (km/h)'
    ])

# --- Helper Functions ---
@st.cache_data
def parse_fit_file(uploaded_file):
    fitfile = FitFile(uploaded_file)
    records = []
    laps_timestamps = []
    
    # 1. Extract laps
    for lap in fitfile.get_messages('lap'):
        lap_data = {}
        for data_field in lap:
            lap_data[data_field.name] = data_field.value
        if 'timestamp' in lap_data:
            laps_timestamps.append(lap_data['timestamp'])

    # 2. Extract continuous records
    for record in fitfile.get_messages('record'):
        data = {}
        for data_field in record:
            data[data_field.name] = data_field.value
        records.append(data)
        
    df = pd.DataFrame(records)
    
    # Standardize necessary columns
    if 'speed' in df.columns: df['speed_ms'] = df['speed']
    if 'distance' in df.columns: df['distance_m'] = df['distance']
    if 'altitude' in df.columns: df['altitude_m'] = df['altitude']
    if 'power' in df.columns: df['power_w'] = df['power']
    
    # Extract GPS data (FIT stores semicircles)
    if 'position_lat' in df.columns and 'position_long' in df.columns:
        df['lat'] = df['position_lat'] * (180.0 / (2**31))
        df['lon'] = df['position_long'] * (180.0 / (2**31))
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Determine lap distances for plotting
    lap_distances = []
    for lt in laps_timestamps:
        lap_row = df[df['timestamp'] <= pd.to_datetime(lt)]
        if not lap_row.empty and 'distance_m' in lap_row.columns:
            lap_distances.append(lap_row.iloc[-1]['distance_m'])
            
    # Calculate delta-t
    df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(1.0)
    df.loc[df['dt'] <= 0, 'dt'] = 1.0
    
    df = df.dropna(subset=['speed_ms', 'power_w', 'distance_m']).reset_index(drop=True)
    return df, lap_distances

def calculate_wind_and_ve(df, cda, crr, m, rho, eta, v_wind_kmh, wind_dir_deg):
    v = df['speed_ms'].values
    p = df['power_w'].values
    dt = df['dt'].values
    
    # Calculate relative air speed (v_air)
    v_wind_ms = v_wind_kmh / 3.6
    
    if 'lat' in df.columns and 'lon' in df.columns and v_wind_ms > 0:
        lat_rad = np.radians(df['lat'])
        lon_rad = np.radians(df['lon'])
        dlon = lon_rad.diff()
        lat1 = lat_rad.shift(1)
        lat2 = lat_rad
        
        # Calculate compass heading
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
        bearing = bearing.bfill()
        
        # Headwind component = wind speed * cos(heading - wind direction)
        headwind_comp = v_wind_ms * np.cos(np.radians(bearing - wind_dir_deg))
        v_air = v + headwind_comp
        v_air = np.maximum(v_air, 0.0)
    else:
        v_air = v

    m_eff = m * 1.03 # Rotational mass factor
    
    p_wheel = p * eta
    p_rr = m * 9.81 * crr * v
    
    # Aerodynamic drag power
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

# --- UI Layout ---
st.title("🚴‍♂️ Aero Analyzer (Chung VE Method)")

with st.sidebar:
    st.header("1. Rider & Bike")
    m_fahrer = st.number_input("Rider Weight (kg)", value=85.0)
    m_rad = st.number_input("Bike Weight + Gear (kg)", value=11.0)
    
    st.header("2. Environment")
    crr = st.number_input("Rolling Resistance (Crr)", value=0.0039, format="%.5f", step=0.0001)
    rho = st.number_input("Air Density (kg/m³)", value=1.162, format="%.3f")
    eta = st.number_input("Drivetrain Efficiency", value=0.975, format="%.3f")
    
    st.header("3. Wind (Optional)")
    v_wind_kmh = st.number_input("Wind Speed (km/h)", value=0.0, step=1.0)
    wind_dir_deg = st.number_input("Wind Direction (Origin in deg, N=0, E=90, S=180, W=270)", value=158.0, step=1.0)
    st.caption("Example: SSE wind (South-South-East) is approx. 158°.")

    st.header("4. CdA Tuning")
    cda_slider = st.slider("CdA Value (m²)", min_value=0.2000, max_value=0.5000, value=0.3560, step=0.0001, format="%.4f")

    st.markdown("---")
    
    with st.expander("ℹ️ Crr & Drivetrain Reference"):
        st.markdown("""
        **Rolling Resistance (Crr)**
        *On smooth, high-quality tarmac:*
        - **0.0025 – 0.0030:** High-end TT tires (Tubeless/Latex/TPU)
        - **0.0030 – 0.0035:** Fast race tires (GP5000, Pro One with Tubeless/TPU)
        - **0.0038 – 0.0045:** Fast race tires with standard Butyl tubes
        - **0.0040 – 0.0048:** All-round/Endurance (Tubeless/TPU)
        - **0.0055 – 0.0070:** Training/Puncture protection tires
        *(Add ~0.0010 - 0.0015 for rough/poor asphalt)*

        **Drivetrain Efficiency (η)**
        - **0.975 – 0.980 (97.5-98%):** Clean, freshly waxed chain, straight chainline
        - **0.960 – 0.970 (96-97%):** Standard chain lube, regular riding conditions
        - **1.000 (100%):** Use ONLY if using a hub-based powermeter (e.g., PowerTap)
        """)
        
    st.markdown("---")
    st.caption("""
    **Disclaimer:** 
    This tool provides aerodynamic estimates based on the Virtual Elevation (Chung) method. 
    The accuracy of the derived CdA depends entirely on the precision of your input variables 
    (weight, air density, Crr, and drivetrain efficiency) and the accuracy of your powermeter and GPS device.
    For educational and personal testing purposes only. Tool made by Dr. Andreas Pfeiffer
    """)

m_system = m_fahrer + m_rad

uploaded_file = st.file_uploader("Upload FIT File", type=["fit"])
st.caption("🔒 **Data: Data is only stored temprarily and will be deleted after closing the window.")

if uploaded_file is not None:
    with st.spinner("Parsing FIT file..."):
        df, lap_distances = parse_fit_file(uploaded_file)
    
    if not 'lat' in df.columns:
        st.warning("Warning: No GPS data found in the file. Dynamic wind correction cannot be calculated.")
    
    st.markdown("### Select Segment")
    min_dist, max_dist = float(df['distance_m'].min()), float(df['distance_m'].max())
    range_dist = st.slider("Distance Range (m)", min_value=min_dist, max_value=max_dist, value=(min_dist, max_dist))
    
    df_filtered = df[(df['distance_m'] >= range_dist[0]) & (df['distance_m'] <= range_dist[1])].copy()
    
    if len(df_filtered) > 0:
        df_calc = calculate_wind_and_ve(df_filtered, cda_slider, crr, m_system, rho, eta, v_wind_kmh, wind_dir_deg)
        
        fig = go.Figure()
        
        # Original Elevation Profile
        if 'altitude_m' in df_calc.columns:
            fig.add_trace(go.Scatter(x=df_calc['distance_m'], y=df_calc['altitude_m'], 
                                     mode='lines', name='Barometric Elevation (GPS)',
                                     line=dict(color='gray', width=2)))
            
        # Calculated Virtual Elevation
        fig.add_trace(go.Scatter(x=df_calc['distance_m'], y=df_calc['v_elev'], 
                                 mode='lines', name='Virtual Elevation (Calculated)',
                                 line=dict(color='blue', width=3)))
        
        # Lap Markers
        for i, lap_dist in enumerate(lap_distances):
            if range_dist[0] <= lap_dist <= range_dist[1]:
                fig.add_vline(x=lap_dist, line_width=1.5, line_dash="dash", line_color="green", 
                              annotation_text=f"Lap {i+1}", annotation_position="top right")
        
        fig.update_layout(title="Elevation Profile vs. Virtual Elevation (Chung)",
                          xaxis_title="Distance (m)",
                          yaxis_title="Elevation (m)",
                          hovermode="x unified")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Save & Export Section (Mobile Friendly) ---
        st.markdown("---")
        st.header("💾 Save & Export Results")
        
        # Eingabe und Button untereinander statt nebeneinander in Spalten
        run_name = st.text_input("Name this run / setup", value="", placeholder="z. B. Setup A - Run 1")
        
        if st.button("💾 Save Current CdA", use_container_width=True):
            new_row = pd.DataFrame([{
                'Setup / Run Name': run_name if run_name else f"Run {len(st.session_state.results_df) + 1}",
                'CdA': round(cda_slider, 4),
                'Crr': crr,
                'Weight (kg)': m_system,
                'Air Density': round(rho, 3),
                'Efficiency': eta,
                'Wind (km/h)': v_wind_kmh
            }])
            st.session_state.results_df = pd.concat([st.session_state.results_df, new_row], ignore_index=True)
            st.success("Ergebnis gespeichert!")

        # Tabelle und Download immer anzeigen
        if not st.session_state.results_df.empty:
            st.dataframe(st.session_state.results_df, use_container_width=True)
            
            csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='cda_test_results.csv',
                mime='text/csv',
                use_container_width=True
            )
        else:
            st.caption("Click 'Save Current CdA', to add this calculation to the table")
