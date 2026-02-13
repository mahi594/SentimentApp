import streamlit as st
import pickle

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")

st.title("💬 Twitter Sentiment Analysis")

tweet = st.text_area("Enter Tweet:")

if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter some text!")
    else:
        tweet_vec = vectorizer.transform([tweet])
        prediction = model.predict(tweet_vec)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
