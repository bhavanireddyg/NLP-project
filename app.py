import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re
from googleapiclient.discovery import build
from googletrans import Translator
from langdetect import detect
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load model and tokenizer
model = tf.keras.models.load_model("model/toxic_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

translator = Translator()

# Function to preprocess text
def preprocess(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    text = text.lower()
    return text

# Detect language and translate to English
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != 'en':
            translated = translator.translate(text, dest='en')
            return translated.text
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Classify comment
def classify_comment(comment):
    translated = translate_to_english(comment)
    cleaned = preprocess(translated)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=150)
    prediction = model.predict(padded)[0][0]
    label = "Toxic ☠️" if prediction >= 0.5 else "Not Toxic ✅"
    return label, prediction

# Extract video ID from YouTube URL
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

# Fetch YouTube comments
def get_youtube_comments(video_id, max_comments=20):
    api_key = "AIzaSyDcEO76-CcCygH5p2_emzsrIkhaqtRF_zQ"  # Replace with your API key
    youtube = build("youtube", "v3", developerKey=api_key)

    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    )
    response = request.execute()

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

# Streamlit UI
st.title("Multilingual Toxic Comment Classifier Bot 🤖")

# Manual input
st.header("🌤 Classify a Single Comment")
user_input = st.text_area("Enter a comment:")
if st.button("Classify Comment"):
    if user_input:
        label, confidence = classify_comment(user_input)
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter a comment.")

# YouTube input
st.header("📺 Classify Comments from a YouTube Video")
video_url = st.text_input("Enter YouTube video link:")
if st.button("Fetch & Classify Comments"):
    if video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            try:
                comments = get_youtube_comments(video_id, max_comments=10)
                results = []

                for comment in comments:
                    label, conf = classify_comment(comment)
                    results.append({
                        "Comment": comment,
                        "Prediction": label,
                        "Confidence": round(conf, 2)
                    })

                df = pd.DataFrame(results)

                st.subheader("📊 Classified Comments")
                st.dataframe(df)

                st.download_button(
                    label="📥 Download Results as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="classified_comments.csv",
                    mime="text/csv"
                )

                # Dashboard - Charts
                st.header("📊 Comment Analysis Dashboard")

                st.subheader("🧮 Toxic vs Non-Toxic Count")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="Prediction", palette="coolwarm", ax=ax)
                st.pyplot(fig)

                st.subheader("🥧 Toxic vs Non-Toxic Pie Chart")
                pie_data = df['Prediction'].value_counts()
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%',
                           startangle=90, colors=["#FF6B6B", "#6BCB77"])
                ax_pie.axis('equal')
                st.pyplot(fig_pie)

                st.subheader("📶 Confidence Score Distribution")
                fig2, ax2 = plt.subplots()
                sns.histplot(df["Confidence"], kde=True, bins=10, color="green", ax=ax2)
                st.pyplot(fig2)

                st.subheader("☁️ WordCloud of Toxic Comments")
                toxic_comments = " ".join(df[df["Prediction"] == "Toxic ☠️"]["Comment"])
                if toxic_comments:
                    wc = WordCloud(width=800, height=400, background_color="white").generate(toxic_comments)
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    ax3.imshow(wc, interpolation="bilinear")
                    ax3.axis("off")
                    st.pyplot(fig3)
                else:
                    st.info("No toxic comments found to generate WordCloud.")

            except Exception as e:
                st.error(f"Failed to fetch comments: {e}")
        else:
            st.error("Invalid YouTube URL.")
