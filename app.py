import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re

# -------------------- CLEANING FUNCTION --------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Model Info")
st.sidebar.write("TF-IDF + Naive Bayes / Logistic Regression")
st.sidebar.write("Detects Fake vs Real News")

# -------------------- TITLE --------------------
st.title("📰 Fake News Detection System")
st.write("Analyze news content and detect whether it is Fake or Real")

# -------------------- INPUT --------------------
st.info("Tip: Use full news paragraphs (2–3 sentences) for better accuracy.")

msg = st.text_area("📝 Enter news text")

uploaded_file = st.file_uploader("Or upload a .txt file")

if uploaded_file is not None:
    msg = uploaded_file.read().decode("utf-8")

# -------------------- PREDICTION --------------------
if st.button("Check"):
    if msg.strip() == "":
        st.warning("⚠ Please enter some text")
    else:
        msg_clean = clean_text(msg)
        vec = vectorizer.transform([msg_clean])

        prob = model.predict_proba(vec)
        fake_prob = prob[0][1]
        real_prob = prob[0][0]

        # Progress
        st.progress(float(fake_prob))
        st.caption("Fake probability meter")

        # Confidence
        st.write(f"Fake: {fake_prob*100:.2f}% | Real: {real_prob*100:.2f}%")

        # Result (improved threshold)
        if fake_prob > 0.7:
            st.error("🚫 Fake News")
        else:
            st.success("✅ Real News")

        # -------------------- IMPORTANT WORDS --------------------
        st.subheader("🔍 Important words in prediction:")
        feature_names = vectorizer.get_feature_names_out()
        vec_array = vec.toarray()[0]

        top_indices = vec_array.argsort()[-10:]

        for i in top_indices:
            if vec_array[i] > 0:
                st.write(feature_names[i])

# -------------------- MODEL PERFORMANCE --------------------
st.subheader("📊 Model Performance")

try:
    acc = pickle.load(open("accuracy.pkl", "rb"))
    cm = pickle.load(open("cm.pkl", "rb"))

    st.write(f"Model Accuracy: {acc:.2f}")

    # Confusion Matrix Plot
    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticklabels(["Real", "Fake"])

    # Add numbers
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center")

    st.pyplot(fig)

    # Breakdown
    st.write("### Breakdown")
    st.write(f"True Negative (Real→Real): {cm[0][0]}")
    st.write(f"False Positive (Real→Fake): {cm[0][1]}")
    st.write(f"False Negative (Fake→Real): {cm[1][0]}")
    st.write(f"True Positive (Fake→Fake): {cm[1][1]}")

except:
    st.warning("⚠ Train the model to see performance metrics")

# -------------------- FOOTER --------------------
st.write("---")
st.write("Developed by Samiksha Chougule")