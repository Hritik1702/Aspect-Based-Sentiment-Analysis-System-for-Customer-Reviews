# Generated from: sentiment_analysis.ipynb
# Converted at: 2025-12-29T06:00:04.138Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ## import lib


import re
import nltk
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ## NLTK download


import os
import nltk

nltk.data.path.append(os.path.expanduser("~/nltk_data"))

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)


# ## importing dataset


DATA_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv"

df = pd.read_csv(DATA_URL)
df = df.rename(columns={"text": "review_text", "stars": "sentiment"})
df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x > 3 else 0)

# ## text processing


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["review_text"].apply(clean_text)


# ## training model


@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = train_model()

# ## aspect defination


ASPECTS = {
    "food": ["food", "taste", "dish", "meal", "flavor"],
    "service": ["service", "staff", "waiter", "waitress"],
    "price": ["price", "cost", "expensive", "cheap", "value"],
    "ambience": ["ambience", "atmosphere", "environment", "place"],
    "delivery": ["delivery", "late", "delay", "time"]
}

def aspect_based_sentiment(review):
    review = review.lower()
    sentences = nltk.sent_tokenize(review)

    aspect_sentiments = {}

    for aspect, keywords in ASPECTS.items():
        scores = []

        for sentence in sentences:
            if any(word in sentence for word in keywords):
                polarity = TextBlob(sentence).sentiment.polarity
                scores.append(polarity)

        if scores:
            avg_score = sum(scores) / len(scores)

            if avg_score > 0.1:
                sentiment = "Positive"
            elif avg_score < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            aspect_sentiments[aspect] = sentiment

    return aspect_sentiments

# ## streamline UI


st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("Sentiment & Aspect-Based Analysis")
st.write("Analyze customer reviews using NLP and Machine Learning")

user_input = st.text_area("Enter a customer review:")

if st.button("Analyze Review"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Overall Sentiment
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        st.subheader("Overall Sentiment")
        if pred == 1:
            st.success("Positive")
        else:
            st.error("Negative")

        # Aspect-Based Sentiment
        st.subheader("Aspect-Based Sentiment")
        aspects = aspect_based_sentiment(user_input)

        if aspects:
            for aspect, sentiment in aspects.items():
                st.write(f"**{aspect.capitalize()}** : {sentiment}")
        else:
            st.write("No specific aspects detected.")

st.markdown("---")
st.caption("Built using TF-IDF, Logistic Regression & NLP")