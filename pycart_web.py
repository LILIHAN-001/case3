import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import *
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the trained model
model = load_model('best_model')  # 加载训练好的ET模型

# Define the feature options
PROM_type_options = {
    0: 'PROM>18h',  # 发作性
    1: 'PROM<18h'  # 持续性
}
MRI_type_options = {
    0: 'No MRI',  # 发作性
    1: 'With MRI'  # 持续性

}

# Streamlit UI
st.title("Electrical Cardioversion Predictor")  # 电复律预测器

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

# Systolic BP input
CRP = st.sidebar.number_input("CRP:", min_value=0, max_value=50, value=1)  # 收缩压输入框

# Age input
IL6 = st.sidebar.number_input("IL6:", min_value=0, max_value=800, value=8)  # 年龄输入框

# AST input
S100B = st.sidebar.number_input("S100B:", min_value=0, max_value=500, value=1)  # AST输入框

# AtrialFibrillationType input
PROM_type = st.sidebar.selectbox("PROM_type:", options=list(PROM_type_options.keys()), format_func=lambda x: PROM_type_options[x])  # 心房颤动类型选择框

MRI_type = st.sidebar.selectbox("MRI_type:", options=list(MRI_type_options.keys()), format_func=lambda x: MRI_type_options[x])  # 心房颤动类型选择框

# Process the input and make a prediction
feature_values = [PROM, S100B, IL6, MRI, CRP ]  # 收集所有输入的特征
# features = np.array([feature_values])  # 转换为NumPy数组
features = pd.DataFrame([feature_values],columns= ["PROM", "S100B", "IL6", "MRI", "CRP" ] ) 

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = predict_model(model, raw_score = True, data=features)["prediction_label"][0] # 预测电复律结果
    predicted_proba = predict_model(model, raw_score = True, data=features)[["prediction_score_0","prediction_score_1"]].values[0]
    # model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class (0 = No, 1 = Yes):** {predicted_class}")  # 显示预测类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测为电复律治疗
        advice = (
            f"According to our model, you may require electrical cardioversion. "
            f"The probability of needing electrical cardioversion is {probability:.1f}%. "
            "This suggests that you may have a higher risk of requiring this treatment. "
            "I recommend consulting with a cardiologist for further examination and possible treatment options."
        )  # 如果预测为需要电复律，给出相关建议
    else:  # 如果预测为不需要电复律
        advice = (
            f"According to our model, you do not require electrical cardioversion. "
            f"The probability of not needing electrical cardioversion is {probability:.1f}%. "
            "However, it is still important to continue regular monitoring of your heart health. "
            "Please ensure you maintain a healthy lifestyle and seek medical attention if needed."
        )  # 如果预测为不需要电复律，给出相关建议

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'No Electrical Cardioversion': predicted_proba[0],  # 不需要电复律的概率
        'Needs Electrical Cardioversion': predicted_proba[1]  # 需要电复律的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['No Electrical Cardioversion', 'Needs Electrical Cardioversion'], 
                    [sample_prob['No Electrical Cardioversion'], sample_prob['Needs Electrical Cardioversion']], 
                    color=['#4caf50', '#fe346e'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Electrical Cardioversion", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['No Electrical Cardioversion'], sample_prob['Needs Electrical Cardioversion']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt)  # 显示图表
