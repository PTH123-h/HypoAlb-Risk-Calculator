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
# Try to find the model file
model_files = ["xgb_model.pkl", "xgb_model.pkl"] 
loaded_model = None

for file in model_files:
    if os.path.exists(file):
        try:
            with open(file, "rb") as f:
                loaded_model = pickle.load(f)
            # Fix: If model is saved as a list, take the first element
            if isinstance(loaded_model, list):
                loaded_model = loaded_model[0]
            break
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

if loaded_model is None:
    st.error("❌ Model file not found! Please make sure 'xgb_model.pkl' is in the same folder.")
    st.stop()

# =======================================================
# C. User Input (Sidebar) - Fully English
# =======================================================
st.sidebar.header("📋 Clinical Parameters")

def user_input_features():
    # 1. Age
    Age = st.sidebar.number_input("Age (years)", min_value=18, max_value=110, value=75)
    
    # 2. Lab Tests (Using standard units)
    CHE = st.sidebar.number_input("Cholinesterase (CHE, U/L)", min_value=100.0, max_value=20000.0, value=5000.0)
    HCT = st.sidebar.number_input("Hematocrit (HCT, %)", min_value=10.0, max_value=70.0, value=40.0)
    hs_CRP = st.sidebar.number_input("High-sensitivity CRP (mg/L)", min_value=0.0, max_value=300.0, value=10.0)
    AG = st.sidebar.number_input("Anion Gap (AG, mmol/L)", min_value=0.0, max_value=50.0, value=12.0)
    Mg = st.sidebar.number_input("Magnesium (Mg, mmol/L)", min_value=0.0, max_value=5.0, value=0.85)
    ALT = st.sidebar.number_input("ALT (U/L)", min_value=0.0, max_value=500.0, value=25.0)
    INR = st.sidebar.number_input("INR", min_value=0.0, max_value=10.0, value=1.1)

    # DataFrame keys must match your training data EXACTLY
    data = {
        'Mg': Mg,
        'ALT': ALT,
        'AG': AG,
        'CHE': CHE,
        'HCT': HCT,
        'INR': INR,
        'hs_CRP': hs_CRP,
        'Age': Age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display Input Data
st.subheader("Patient Data Confirmation")
st.dataframe(input_df)

# =======================================================
# D. Prediction Logic (With Normalization)
# =======================================================
if st.button("🚀 Calculate Risk"):
    try:
        # 1. Get Raw Probability
        prediction_proba = loaded_model.predict_proba(input_df)
        raw_prob = float(prediction_proba[0][1]) # 原始概率
        
        # 2. Define Your Optimal Threshold (Youden Index)
        threshold = 0.3396 
        
        # 🟢 3. Normalization Calculation (归一化核心代码)
        # 目的：将 threshold 映射为 0.5，符合用户“超过50%即高危”的直觉
        if raw_prob < threshold:
            # 区间 [0, 0.3396] 映射到 [0, 0.5]
            display_prob = (raw_prob / threshold) * 0.5
        else:
            # 区间 [0.3396, 1.0] 映射到 [0.5, 1.0]
            display_prob = 0.5 + ((raw_prob - threshold) / (1 - threshold)) * 0.5
            
        # 4. Display Results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        # 显示的是“校准后”的概率 (看起来更直观)
        col1.metric("Risk Score", f"{display_prob:.1%}")
        col1.progress(display_prob)
        
        # 5. Risk Stratification
        # 现在只需要判断 display_prob 是否大于 0.5 即可
        with col2:
            if display_prob > 0.5:
                st.error("⚠️ **High Risk Group**")
                st.markdown(f"The predicted risk score is **High (>50%)**.")
                st.markdown("**Recommendation:** Clinical nutritional intervention is suggested.")
            else:
                st.success("✅ **Low Risk Group**")
                st.markdown(f"The predicted risk score is **Low (<50%)**.")
        
        # 6. Technical Note (为了严谨，加一行小字解释)
        st.caption(f"Note: The displayed Risk Score is a calibrated probability based on the optimal model threshold ({threshold}). Scores >50% indicate high risk.")
                
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Tip: Check if the column names in 'user_data' match your XGBoost model features.")

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
