import streamlit as st
import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
import sys

# Recreate SentimentExtractor used in training so joblib can unpickle the pipeline
class SentimentExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scores = [self.analyzer.polarity_scores(text)["compound"] for text in X]
        return np.array(scores).reshape(-1, 1)

# Ensure the unpickler can find the class under the '__main__' namespace
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "SentimentExtractor", SentimentExtractor)

# Load trained pipeline model
MODEL_PATH = r"C:\Users\Ayush Ahlawat\OneDrive\Documents\Public Comment Analysis\public-comment-analysis\src\models\stance_model_v2.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Policy Stance Classifier", page_icon="🧠")

st.title("🧠 Policy Stance Classification System")
st.write("Enter any policy-related public comment to detect stance (Support / Oppose / Neutral).")

text = st.text_area("Enter comment below:", height=200)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a comment before predicting.")
    else:
        pred = model.predict([text])[0]
        conf = model.predict_proba([text]).max() * 100

        st.subheader("Prediction Result")
        st.write(f"### 🎯 **Stance:** {pred}")
        st.write(f"### 🔐 **Confidence:** {conf:.2f}%")
