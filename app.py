import streamlit as st
import pandas as pd
import pickle
import os

# =======================================================
# A. Page Configuration
# =======================================================
st.set_page_config(
    page_title="Hypoalbuminemia Risk Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("🏥 Risk Prediction Model for Hypoalbuminemia")
st.markdown("### in Elderly Patients with AECOPD")
st.markdown("---")
st.info(
    "💡 **Instructions:** Please input the patient's clinical parameters in the sidebar "
    "to estimate the risk of hypoalbuminemia based on the XGBoost machine learning model."
)

# =======================================================
# B. Load Model (Auto-detection)
# =======================================================
model_files = ["xgb_model.pkl", "xgb_model.pkl"] 
loaded_model = None

for file in model_files:
    if os.path.exists(file):
        try:
            with open(file, "rb") as f:
                loaded_model = pickle.load(f)
            if isinstance(loaded_model, list):
                loaded_model = loaded_model[0]
            break
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

if loaded_model is None:
    st.error("❌ Model file not found! Please make sure 'xgb_model.pkl' is in the same folder.")
    st.stop()

# =======================================================
# C. User Input (Sidebar)
# =======================================================
st.sidebar.header("📋 Clinical Parameters")

def user_input_features():
    # 1. Age
    Age = st.sidebar.number_input("Age (years)", min_value=18, max_value=110, value=75)
    
    # 2. Lab Tests
    CHE = st.sidebar.number_input("Cholinesterase (CHE, U/L)", min_value=100.0, max_value=20000.0, value=5000.0)
    HCT = st.sidebar.number_input("Hematocrit (HCT, %)", min_value=10.0, max_value=70.0, value=40.0)
    hs_CRP = st.sidebar.number_input("High-sensitivity CRP (mg/L)", min_value=0.0, max_value=300.0, value=10.0)
    AG = st.sidebar.number_input("Anion Gap (AG, mmol/L)", min_value=0.0, max_value=50.0, value=12.0)
    Mg = st.sidebar.number_input("Magnesium (Mg, mmol/L)", min_value=0.0, max_value=5.0, value=0.85)
    ALT = st.sidebar.number_input("ALT (U/L)", min_value=0.0, max_value=500.0, value=25.0)
    INR = st.sidebar.number_input("INR", min_value=0.0, max_value=10.0, value=1.1)

    data = {
        'Mg': Mg, 'ALT': ALT, 'AG': AG, 'CHE': CHE, 
        'HCT': HCT, 'INR': INR, 'hs_CRP': hs_CRP, 'Age': Age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display Input Data
st.subheader("Patient Data Confirmation")
st.dataframe(input_df)

# =======================================================
# D. Prediction Logic (Dual Display)
# =======================================================
if st.button("🚀 Calculate Risk"):
    try:
        # 1. Get Raw Probability (Matches SHAP)
        prediction_proba = loaded_model.predict_proba(input_df)
        raw_prob = float(prediction_proba[0][1])
        
        # 2. Define Threshold
        threshold = 0.3396 
        
        # 3. Calculate Normalized Score (Matches Red/Green Alert)
        if raw_prob < threshold:
            display_prob = (raw_prob / threshold) * 0.5
        else:
            display_prob = 0.5 + ((raw_prob - threshold) / (1 - threshold)) * 0.5
            
        # 4. Display Results (Side-by-Side)
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 左边：显示原始概率 (给审稿人看，对应 SHAP)
            st.metric("Raw Probability", f"{raw_prob:.2%}")
            st.caption(f"Model Probability (Threshold: {threshold})")
            
        with col2:
            # 右边：显示校准分数 (给医生看，直观理解)
            st.metric("Clinical Risk Score", f"{display_prob:.1%}")
            st.progress(display_prob)

        # 5. Risk Alert (Based on the Score)
        st.markdown("") # Add some space
        if display_prob > 0.5:
            st.error("⚠️ **High Risk Group**")
            st.markdown(f"The patient is classified as **High Risk**.")
            st.markdown("**Recommendation:** Clinical nutritional intervention is suggested.")
        else:
            st.success("✅ **Low Risk Group**")
            st.markdown(f"The patient is classified as **Low Risk**.")
            
        # 6. Explanation Note
        st.info(f"Note: 'Raw Probability' is the direct output from the model. 'Clinical Risk Score' is a calibrated value where >50% indicates High Risk (corresponding to Raw Probability > {threshold}).")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 12px;'>
        <b>© 2026 AECOPD Research Group. All rights reserved.</b><br>
        For Research Use Only | Not for Clinical Diagnosis
    </div>
    """, 
    unsafe_allow_html=True
)
