import streamlit as st
import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
import sys

# =============================================================
# 1) RECREATE SentimentExtractor EXACTLY LIKE TRAINING PIPELINE
# =============================================================

class SentimentExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Must return 4 features per text sample:
        [neg, neu, pos, compound]
        """
        out = []
        for text in X:
            s = self.vader.polarity_scores(text)
            out.append([s["neg"], s["neu"], s["pos"], s["compound"]])
        return np.array(out)

# Make sure joblib finds the class under __main__
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "SentimentExtractor", SentimentExtractor)

# =============================================================
# 2) LOAD TRAINED MODEL
# =============================================================

MODEL_PATH = r"C:\Users\Ayush Ahlawat\OneDrive\Documents\Public Comment Analysis\public-comment-analysis\src\models\stance_model_balanced.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# =============================================================
# 3) STREAMLIT UI
# =============================================================
st.set_page_config(page_title="Policy Stance Classifier", page_icon="🧠", layout="wide")

st.title("🧠 Policy Stance Classification System")
st.write("This model predicts whether a public policy comment expresses **Support**, **Oppose**, or **Neutral** stance.")

# Text input
text = st.text_area("📝 Enter a policy-related comment:", height=200)

# =============================================================
# 4) PREDICTION
# =============================================================

if st.button("🔍 Predict Stance"):
    if not text.strip():
        st.warning("⚠️ Please enter a comment before predicting.")
    else:
        pred = model.predict([text])[0]
        conf = model.predict_proba([text]).max() * 100
        
        # UI Output
        st.subheader("📌 Prediction Result")
        
        if pred == "for":
            st.success(f"### ✔️ **Stance: SUPPORTING**")
        elif pred == "against":
            st.error(f"### ❌ **Stance: OPPOSING**")
        else:
            st.info(f"### ⚪ **Stance: NEUTRAL**")

        st.write(f"### 🔐 Confidence Score: **{conf:.2f}%**")

        # Optional: Show raw sentiment features
        st.write("---")
        vader = SentimentIntensityAnalyzer()
        s = vader.polarity_scores(text)
        st.write("### 🧾 Sentiment Breakdown (VADER)")
        st.json(s)

st.write("---")
st.caption("Developed by Ayush • Policy Stance NLP Model • Streamlit Deployment")
