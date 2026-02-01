import streamlit as st
import pandas as pd
import pickle
import os

# =======================================================
# A. Page Configuration (Compact Mode)
# =======================================================
st.set_page_config(
    page_title="Risk Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 1. Compact Header
# 使用两列布局：左边放标题，右边放一个小小的关于按钮
col_header, col_help = st.columns([0.85, 0.15])
with col_header:
    st.title("🏥 Hypoalbuminemia Risk Model")
    st.caption("For Research Use Only | Target: Elderly Patients with AECOPD")

with col_help:
    with st.popover("ℹ️ Info"):
        st.markdown("Input clinical parameters in the sidebar to estimate risk.")

st.markdown("---")

# =======================================================
# B. Load Model
# =======================================================
model_files = ["xgb_model.pkl", "xgb_model.pkl"] 
loaded_model = None
for file in model_files:
    if os.path.exists(file):
        try:
            with open(file, "rb") as f:
                loaded_model = pickle.load(f)
            if isinstance(loaded_model, list): loaded_model = loaded_model[0]
            break
        except: pass

if loaded_model is None:
    st.error("Model not found.")
    st.stop()

# =======================================================
# C. User Input (Sidebar)
# =======================================================
st.sidebar.header("📋 Patient Data")

def user_input_features():
    Age = st.sidebar.number_input("Age (years)", 18, 110, 75)
    CHE = st.sidebar.number_input("Cholinesterase (U/L)", 100.0, 20000.0, 5000.0)
    HCT = st.sidebar.number_input("Hematocrit (%)", 10.0, 70.0, 40.0)
    hs_CRP = st.sidebar.number_input("hs-CRP (mg/L)", 0.0, 300.0, 10.0)
    AG = st.sidebar.number_input("Anion Gap (mmol/L)", 0.0, 50.0, 12.0)
    Mg = st.sidebar.number_input("Magnesium (mmol/L)", 0.0, 5.0, 0.85)
    ALT = st.sidebar.number_input("ALT (U/L)", 0.0, 500.0, 25.0)
    INR = st.sidebar.number_input("INR", 0.0, 10.0, 1.1)

    return pd.DataFrame({
        'Mg': Mg, 'ALT': ALT, 'AG': AG, 'CHE': CHE, 
        'HCT': HCT, 'INR': INR, 'hs_CRP': hs_CRP, 'Age': Age
    }, index=[0])

input_df = user_input_features()

# 🔴 移除了中间冗余的 Dataframe 显示，节省大量垂直空间！

# =======================================================
# D. Prediction Logic (Compact Display)
# =======================================================
# 按钮上方加一点点空隙，不如直接放按钮
if st.button("🚀 Calculate Risk", type="primary", use_container_width=True):
    try:
        prediction_proba = loaded_model.predict_proba(input_df)
        raw_prob = float(prediction_proba[0][1])
        threshold = 0.3396 
        
        # Normalization
        if raw_prob < threshold:
            display_prob = (raw_prob / threshold) * 0.5
        else:
            display_prob = 0.5 + ((raw_prob - threshold) / (1 - threshold)) * 0.5
            
        # --- Compact Result Section ---
        # 使用容器把结果包起来，显得更紧凑
        with st.container(border=True):
            c1, c2 = st.columns(2)
            
            with c1:
                # 原始概率 (Raw)
                st.metric("Raw Probability", f"{raw_prob:.2%}", help=f"Original Model Output\nThreshold: {threshold}")
                
            with c2:
                # 评分 (Score)
                st.metric("Clinical Risk Score", f"{display_prob:.1%}", help=">50% indicates High Risk")
                st.progress(display_prob)

            # --- Compact Alert Box ---
            # 合并了状态和建议，不再分行写
            if display_prob > 0.5:
                st.error(f"⚠️ **High Risk** (Raw > {threshold})\n\n**Suggestion:** Clinical nutritional intervention is recommended.")
            else:
                st.success(f"✅ **Low Risk** (Raw < {threshold})\n\n**Suggestion:** Routine monitoring.")
            
            # 极简的注脚
            st.caption(f"Note: Risk Score >50% corresponds to Raw Probability > {threshold}.")

    except Exception as e:
        st.error(f"Error: {e}")

# =======================================================
# Footer (Minimalist)
# =======================================================
st.markdown(
    """
    <div style='position: fixed; bottom: 0; left: 0; width: 100%; background-color: white; text-align: center; color: grey; font-size: 10px; padding: 5px; border-top: 1px solid #eee;'>
        © 2026 AECOPD Research Group | Research Use Only
    </div>
    """, 
    unsafe_allow_html=True
)
