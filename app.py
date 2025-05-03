# crop_recommender.py - Updated with NPK sliders and Soil Moisture input
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide", page_title="🌱 AgroSense-AI: The Smart Crop Compatibility Engine", page_icon="🌾")

# --- Data Loading and Model Training ---
@st.cache_resource
def load_data_and_train():
    df = pd.read_csv("Crop_Recommendation.csv")

    df = df.rename(columns={
        'Nitrogen': 'N',
        'Phosphorus': 'P',
        'Potassium': 'K',
        'pH_Value': 'pH',
        'Soil_Moisture': 'Soil_Moisture'  # Ensure this column is in the CSV
    })

    le = LabelEncoder()
    df['Crop'] = le.fit_transform(df['Crop'])
    model = RandomForestClassifier()
    model.fit(df.drop(['Crop', 'Rainfall'], axis=1), df['Crop'])  # Exclude Rainfall during model training

    crop_stats = {}
    for crop in le.classes_:
        crop_data = df[df['Crop'] == le.transform([crop])[0]]
        stats = {
            'N_avg': crop_data['N'].mean(),
            'P_avg': crop_data['P'].mean(),
            'K_avg': crop_data['K'].mean(),
            'N_range': (crop_data['N'].min(), crop_data['N'].max()),
            'P_range': (crop_data['P'].min(), crop_data['P'].max()),
            'K_range': (crop_data['K'].min(), crop_data['K'].max()),
            'Temperature': (crop_data['Temperature'].min(), crop_data['Temperature'].max()),
            'Humidity': (crop_data['Humidity'].min(), crop_data['Humidity'].max()),
            'pH': (crop_data['pH'].min(), crop_data['pH'].max()),
            'Soil_Moisture': (crop_data['Soil_Moisture'].min(), crop_data['Soil_Moisture'].max())
        }
        crop_stats[crop] = stats

    return model, le, crop_stats

model, le, crop_stats = load_data_and_train()
all_crops = sorted(le.classes_)

st.title("🌾 AgroSense-AI: The Smart Crop Compatibility Engine")
st.markdown("""
*Customize soil nutrient values (N-P-K) and environmental factors*  
*Get personalized crop recommendations based on your inputs*
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🌿 Crop Selection")
    selected_crop = st.selectbox("Choose your crop", all_crops)

    st.header("🧪 Soil Nutrients")
    # Get min/max ranges from the dataset for sliders
    n_min, n_max = 0, 300  # You can adjust these based on your data
    p_min, p_max = 0, 300
    k_min, k_max = 0, 300
    
    # Use the crop's average values as default but allow customization
    n_value = st.slider("Nitrogen (N) ppm", 
                        n_min, n_max, 
                        int(round(crop_stats[selected_crop]['N_avg'], 0)),  # Round and convert to int
                        step=1,  # Ensure step is integer
                        help=f"Typical range for {selected_crop}: {crop_stats[selected_crop]['N_range'][0]:.1f}-{crop_stats[selected_crop]['N_range'][1]:.1f}")
    
    p_value = st.slider("Phosphorus (P) ppm", 
                        p_min, p_max, 
                        int(round(crop_stats[selected_crop]['P_avg'], 0)),  # Round and convert to int
                        step=1,  # Ensure step is integer
                        help=f"Typical range for {selected_crop}: {crop_stats[selected_crop]['P_range'][0]:.1f}-{crop_stats[selected_crop]['P_range'][1]:.1f}")
    
    k_value = st.slider("Potassium (K) ppm", 
                        k_min, k_max, 
                        int(round(crop_stats[selected_crop]['K_avg'], 0)),  # Round and convert to int
                        step=1,  # Ensure step is integer
                        help=f"Typical range for {selected_crop}: {crop_stats[selected_crop]['K_range'][0]:.1f}-{crop_stats[selected_crop]['K_range'][1]:.1f}")

    st.header("🌦️ Environmental Factors")
    temp = st.slider("Temperature (°C)", 8.8, 43.7, 25.0, step=0.1)
    humidity = st.slider("Humidity (%)", 14.3, 99.9, 50.0, step=0.1)
    ph = st.slider("Soil pH Level", 3.5, 9.9, 6.5, step=0.1)
    soil_moisture = st.slider("Soil Moisture (%)", 10.0, 90.0, 50.0, step=0.1)

# --- Analysis Logic ---
if st.button("🧑‍🌾 Analyze Growing Conditions", type="primary"):
    st.header("🔍 Analysis Results")

    analysis_data = {
        'N': n_value,
        'P': p_value,
        'K': k_value,
        'Temperature': temp,
        'Humidity': humidity,
        'pH': ph,
        'Soil_Moisture': soil_moisture
    }

    requirements = crop_stats[selected_crop]
    violations = []
    
    # Check all parameters including NPK
    for param in ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Soil_Moisture']:
        if param in ['N', 'P', 'K']:
            min_val, max_val = requirements[f'{param}_range']
        else:
            min_val, max_val = requirements[param]
            
        if not (min_val <= analysis_data[param] <= max_val):
            violations.append(f"{param}: {analysis_data[param]} (requires {min_val}-{max_val})")

    if not violations:
        st.success(f"✅ Excellent conditions for {selected_crop}!")
        st.balloons()
    else:
        st.error(f"⚠️ Potential challenges for {selected_crop}:")
        for issue in violations:
            st.write(f"- {issue}")
        st.warning("Consider adjusting your inputs or choosing a different crop")

    # Alternative crops based on all parameters including NPK
    st.subheader("🌱 Alternative Suitable Crops")
    suitable_crops = []
    for crop in all_crops:
        reqs = crop_stats[crop]
        if all(
            reqs[f'{param}_range'][0] <= analysis_data[param] <= reqs[f'{param}_range'][1] if param in ['N', 'P', 'K'] 
            else reqs[param][0] <= analysis_data[param] <= reqs[param][1]
            for param in ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Soil_Moisture']
        ):
            suitable_crops.append((crop, reqs['N_avg'], reqs['P_avg'], reqs['K_avg']))

    if suitable_crops:
        cols = st.columns(3)
        for i, (crop, n, p, k) in enumerate(suitable_crops):
            with cols[i % 3]:
                with st.expander(f"**{crop}**", expanded=True):
                    st.metric("N-P-K", f"{n:.1f}-{p:.1f}-{k:.1f}")
                    st.caption(f"Temp: {crop_stats[crop]['Temperature'][0]}–{crop_stats[crop]['Temperature'][1]}°C")
                    st.caption(f"pH: {crop_stats[crop]['pH'][0]}–{crop_stats[crop]['pH'][1]}")
                    st.caption(f"Soil Moisture: {crop_stats[crop]['Soil_Moisture'][0]}–{crop_stats[crop]['Soil_Moisture'][1]}%")
    else:
        st.warning("No crops perfectly match these conditions.")

# --- Crop Requirement Table ---
with st.expander("📊 Crop Requirement Table", expanded=False):
    summary_data = []
    for crop in all_crops:
        stats = crop_stats[crop]
        summary_data.append({
            'Crop': crop,
            'N Range (ppm)': f"{stats['N_range'][0]:.1f}-{stats['N_range'][1]:.1f}",
            'P Range (ppm)': f"{stats['P_range'][0]:.1f}-{stats['P_range'][1]:.1f}",
            'K Range (ppm)': f"{stats['K_range'][0]:.1f}-{stats['K_range'][1]:.1f}",
            'Temp (°C)': f"{stats['Temperature'][0]}–{stats['Temperature'][1]}",
            'Humidity (%)': f"{stats['Humidity'][0]}–{stats['Humidity'][1]}",
            'pH Range': f"{stats['pH'][0]}–{stats['pH'][1]}",
            'Soil Moisture (%)': f"{stats['Soil_Moisture'][0]}–{stats['Soil_Moisture'][1]}"
        })

    st.dataframe(
        pd.DataFrame(summary_data),
        height=500,
        use_container_width=True,
        column_config={
            "N Range (ppm)": st.column_config.NumberColumn(
                "Nitrogen Range",
                help="Required Nitrogen range in ppm"
            ),
            "P Range (ppm)": st.column_config.NumberColumn(
                "Phosphorus Range",
                help="Required Phosphorus range in ppm"
            ),
            "K Range (ppm)": st.column_config.NumberColumn(
                "Potassium Range",
                help="Required Potassium range in ppm"
            )
        }
    )
