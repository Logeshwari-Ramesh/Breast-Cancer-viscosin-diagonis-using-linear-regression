import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model + accuracy
model, acc = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Cancer Predictor", layout="centered")

st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #FF4B4B;
}
.subtitle {
    text-align: center;
    color: #AAAAAA;
}
.card {
    background-color: #161A21;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Breast Cancer Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML model to detect cancer</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.info("Created by Logu\nAI & Data Science\n2026")
st.sidebar.markdown("---")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Accuracy Score:", round(acc, 4))   # ✅ FINAL OUTPUT

# Input
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Enter Tumor Details")

radius = st.slider("Mean Radius", 5.0, 30.0, 15.0)
texture = st.slider("Mean Texture", 5.0, 40.0, 20.0)
perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 80.0)
area = st.slider("Mean Area", 200.0, 2500.0, 800.0)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if st.button("Predict"):
    features = [radius, texture, perimeter, area]

    result = model.predict([features])
    prob = model.predict_proba([features])

    if result[0] == 1:
        st.success("Benign (Non-cancerous)")
    else:
        st.error("Malignant (Cancerous)")

    st.write("Cancer Probability:", round(prob[0][0]*100, 2), "%")

    # Graph
    df = pd.DataFrame({
        "Feature": ["Radius", "Texture", "Perimeter", "Area"],
        "Value": [radius, texture, perimeter, area]
    })

    fig, ax = plt.subplots()
    ax.bar(df["Feature"], df["Value"])
    ax.set_xlabel("Features")
    ax.set_ylabel("Values")
    ax.set_title("Input Features")

    st.pyplot(fig)

# Footer
st.markdown("---")
st.subheader("About Project")
st.write("This application uses Machine Learning to classify tumors.")
st.write("Built using Logistic Regression and Streamlit.")

st.markdown("<div class='footer'>Created by Logu | AI & Data Science | 2026</div>", unsafe_allow_html=True))