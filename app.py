import streamlit as st
import pandas as pd
import pickle
import os

# =======================================================
# 1. 页面基础设置 (Page Config)
# =======================================================
st.set_page_config(
    page_title="AECOPD Risk Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 🎨【关键修改】注入 CSS 样式，强制减少顶部留白，让截图更紧凑好看
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1 {
            font-size: 2.2rem !important;
            margin-bottom: 0rem !important;
        }
        .stAlert {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# =======================================================
# 2. 加载模型 (Load Model)
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
    st.error("❌ Model missing. Please check file path.")
    st.stop()

# =======================================================
# 3. 侧边栏输入 (Sidebar)
# =======================================================
with st.sidebar:
    st.header("📋 Patient Parameters")
    st.markdown("---")
    
    # 使用紧凑的输入框
    Age = st.number_input("Age (years)", 18, 110, 75)
    
    c1, c2 = st.columns(2)
    with c1:
        CHE = st.number_input("CHE (U/L)", 100.0, 20000.0, 5000.0)
        HCT = st.number_input("HCT (%)", 10.0, 70.0, 40.0)
        AG = st.number_input("AG (mmol/L)", 0.0, 50.0, 12.0)
        ALT = st.number_input("ALT (U/L)", 0.0, 500.0, 25.0)
    with c2:
        hs_CRP = st.number_input("hs-CRP (mg/L)", 0.0, 300.0, 10.0)
        Mg = st.number_input("Mg (mmol/L)", 0.0, 5.0, 0.85)
        INR = st.number_input("INR", 0.0, 10.0, 1.1)
        # 占位符，保持对齐
        st.write("") 

    input_df = pd.DataFrame({
        'Mg': Mg, 'ALT': ALT, 'AG': AG, 'CHE': CHE, 
        'HCT': HCT, 'INR': INR, 'hs_CRP': hs_CRP, 'Age': Age
    }, index=[0])
    
    st.markdown("---")
    st.caption("© 2026 AECOPD Research Group")

# =======================================================
# 4. 主界面布局 (Main Layout)
# =======================================================

# 标题区 (带图标，显眼)
c_logo, c_title = st.columns([0.1, 0.9])
with c_logo:
    st.markdown("# 🏥")
with c_title:
    st.title("Hypoalbuminemia Risk Prediction")
    st.markdown("**Target Population:** Elderly Patients with AECOPD")

# 按钮区 (美化按钮)
st.markdown("") # 加一点点间距
if st.button("🚀 Run Risk Assessment", type="primary", use_container_width=True):
    
    # --- 预测逻辑 ---
    prediction_proba = loaded_model.predict_proba(input_df)
    raw_prob = float(prediction_proba[0][1])
    threshold = 0.3396 
    
    # 归一化逻辑
    if raw_prob < threshold:
        display_prob = (raw_prob / threshold) * 0.5
    else:
        display_prob = 0.5 + ((raw_prob - threshold) / (1 - threshold)) * 0.5

    # --- 结果展示区 (卡片式设计) ---
    st.markdown("### 📊 Assessment Result")
    
    # 使用边框容器，像一张报告单
    with st.container(border=True):
        
        # 第一排：两个核心指标
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            # 原始概率 (给审稿人/对应SHAP)
            st.metric(
                label="Raw Probability (Model)", 
                value=f"{raw_prob:.2%}",
                delta="> 33.96% Threshold" if raw_prob > threshold else None,
                delta_color="inverse",
                help="Direct output from the XGBoost model."
            )
            
        with col_res2:
            # 临床评分 (给医生/红绿灯)
            st.metric(
                label="Clinical Risk Score", 
                value=f"{display_prob:.1%}",
                help="Calibrated score. >50% indicates High Risk."
            )
        
        # 进度条
        st.progress(display_prob)
        
        # 分割线
        st.markdown("---")
        
        # 最终判定 (醒目的提示框)
        if display_prob > 0.5:
            st.error(
                "#### ⚠️ High Risk Detected\n"
                "The patient shows a high probability of hypoalbuminemia.\n\n"
                "**Recommendation:** Early nutritional intervention is strongly suggested."
            )
        else:
            st.success(
                "#### ✅ Low Risk\n"
                "The probability of hypoalbuminemia is low.\n\n"
                "**Recommendation:** Routine monitoring."
            )

        # 底部小字
        st.caption(f"Technical Note: Risk Score >50% aligns with Raw Probability > {threshold} (Youden Index).")

else:
    # 默认状态下的占位提示 (为了让页面不显得空)
    st.info("👈 Please input clinical parameters in the sidebar and click 'Run Risk Assessment'.")
