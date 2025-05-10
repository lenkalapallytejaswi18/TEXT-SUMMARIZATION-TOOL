# 📦 Import required libraries
import streamlit as st
from transformers import pipeline
import re

st.set_page_config(page_title="Smart AI Text Summarizer", layout="centered")
st.title("📝 Smart AI Text Summarizer")
st.write("Summarize long or short texts without errors using advanced NLP models.")

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="google/flan-t5-large",  # OR use "facebook/bart-large-cnn" if you want
        device=0  # Comment/remove if CPU
    )

summarizer = load_model()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

# ✂️ Smart truncate function
def truncate_text(text, max_words=500):
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    return ' '.join(words)

input_text = st.text_area("Enter text to summarize:", height=300)

min_len = st.slider("Minimum summary length (words)", 10, 80, 20)
max_len = st.slider("Maximum summary length (words)", 30, 200, 60)

if st.button("Summarize"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("Summarizing, please wait..."):
            try:
                cleaned_text = clean_text(input_text)
                short_text = truncate_text(cleaned_text, max_words=500)

                prompt_text = "Summarize: " + short_text

                summary_output = summarizer(
                    prompt_text,
                    min_length=min_len,
                    max_length=max_len,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    num_beams=4,
                    early_stopping=True
                )
                summary_text = summary_output[0]['summary_text']

                st.subheader("📋 Summary")
                st.success(summary_text)

            except Exception as e:
                st.error(f"🚨 Error: {e}")
