import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pycaret.regression import *
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
shap.initjs()

def VSpace(px):
    """一个简单的函数，用于在 Streamlit 中创建指定像素的垂直空间"""
    st.markdown(f'<div style="margin-top: {px}px;"></div>', unsafe_allow_html=True)


# Load the trained model
model = load_model('best_model')  # 加载训练好的ET模型

# Define the feature options
FEA1_type_options = {
    0: 'FEA1<52',  
    1: 'FEA1>52'  
}
FEA4_type_options = {
    0: 'No Surgery',  
    1: 'With Surgery'  
}

# Streamlit UI
st.title("Early Diagnosis Predictor")  

image =Image.open("Snipaste_2025-07-01_13-45-35.png")
st.image(image )


# Sidebar for input options
st.sidebar.header("Input Data")  # 侧边栏输入样本数据

FEA5 = st.sidebar.number_input("FEA5:", min_value=0, max_value=50, value=10)  # 收缩压输入框

FEA3 = st.sidebar.number_input("FEA3:", min_value=0, max_value=800, value=80)  # 年龄输入框

FEA2 = st.sidebar.number_input("FEA2:", min_value=0, max_value=500, value=100)  # AST输入框

FEA1 = st.sidebar.selectbox("FEA1_type:", options=list(FEA1_type_options.keys()), format_func=lambda x: FEA1_type_options[x])  # 心房颤动类型选择框

FEA4 = st.sidebar.selectbox("FEA4_type:", options=list(FEA4_type_options.keys()), format_func=lambda x: FEA4_type_options[x])  # 心房颤动类型选择框
# 添加一个 50 像素的垂直空白
VSpace(50)

st.subheader("Process the input and make a prediction")
# Process the input and make a prediction
feature_values = [FEA1, FEA2, FEA3, FEA4, FEA5 ]  # 收集所有输入的特征
features = pd.DataFrame([feature_values],columns= ["FEA1", "FEA2", "FEA3", "FEA4", "FEA5" ] ) 

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = predict_model(model, raw_score = True, data=features)["prediction_label"][0] # 预测结果
    predicted_proba = predict_model(model, raw_score = True, data=features)[["prediction_score_0","prediction_score_1"]].values[0]
    # model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class (0 = No, 1 = Yes):** {predicted_class}")  # 显示预测类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比


    # Visualize the prediction probabilities
    sample_prob = {
        'No Disease Occurrence': predicted_proba[0],  # 疾病不发生的概率
        'Disease Occurrence': predicted_proba[1]  # 疾病发生的概率
    }
    
    VSpace(20)
    # Set figure size
    plt.figure(figsize=(4, 1))  # 设置图形大小
    plt.rc('ytick', labelsize=8) # 设置所有Y轴刻度的字体大小
    plt.rc('xtick', labelsize=8) # 设置所有X轴刻度的字体大小
    # Create bar chart
    bars = plt.barh(['No Disease Occurrence', 'Disease Occurrence'], 
                    [sample_prob['No Disease Occurrence'], sample_prob['Disease Occurrence']], 
                    height=0.6, edgecolor="black", color=['#81abd3','#fcd6d3'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Disease Occurrence", fontsize=12, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=7 )  # 添加X轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['No Disease Occurrence'], sample_prob['Disease Occurrence']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=6, color='black' )  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt, use_container_width=True)  # 显示图表
    

    if predicted_class == 1:  # 如果预测为疾病发生，给出相关建议
        advice = (
            f"**Recommendation:** According to our model, the probability of the disease occurring is {probability:.1f}%, which is considered **High risk** . "
            f"We recommend you discuss these findings with your doctor or a relevant specialist as soon as possible to determine the next steps for care and follow-up."
        )  
    else:  # 如果预测为不需要疾病低风险
        advice = (
            f"**Recommendation:** According to our model, the patient is at **low risk**. "
            f"The probability of the disease **not occurring** is **{probability:.1f}%**. "
            "However, it is still important to continue regular monitoring of your heart health. "
            "Please ensure you maintain a healthy lifestyle and seek medical attention if needed."
        )  

    st.write(advice)  # 显示建议
    
    VSpace(50)

    st.subheader("Feature importance")
    model_estimator = model.named_steps['actual_estimator'] 
    explainer = shap.TreeExplainer(model_estimator)
    shap_values = explainer.shap_values(features)

#     fig, ax = plt.subplots(figsize=(3, 2.5)) 

#     shap.force_plot(
#     explainer.expected_value,
#     shap_values[:,:],
#     features,
#     matplotlib=True
# )
#     plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
#     plt.close(fig)
#     st.image("shap_force_plot.png", use_container_width=True)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0], # shap_values[class_index][idx,:],
        base_values=explainer.expected_value, # explainer.expected_value[class_index]
        data=features.values[0],
        feature_names=features.columns.tolist()
    ) 
)
    plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    st.image("shap_waterfall_plot.png", use_container_width=True)
