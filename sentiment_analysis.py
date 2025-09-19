import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

# ----------------------------
# Setup
# ----------------------------
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load CSV
file_path = r"C:\Users\User\Downloads\niText SMS PLATFORM - Monday Report.csv"
df = pd.read_csv(file_path)

# Define stopwords + extra conjunctions
stop_words = set(stopwords.words('english'))
conjunctions = {
    "2","cs","hi","there","is","a","actually","may","iknow","pls",
    "to","the","ig","honorable","please","ask","updf","one","two",
    "na","and","ni","plz","thing","always","havent"
}
stop_words = stop_words.union(conjunctions)

# ----------------------------
# Text Cleaning
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters + spaces
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

df["Cleaned_Message"] = df["Message"].apply(clean_text)

# Remove duplicates
df = df.drop_duplicates(subset=["Cleaned_Message"]).reset_index(drop=True)

# ----------------------------
# VADER Sentiment
# ----------------------------
sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["VADER_Sentiment"] = df["Cleaned_Message"].apply(vader_sentiment)

# ----------------------------
# Hugging Face Sentiment
# ----------------------------
# Force PyTorch framework to avoid TensorFlow/Keras issues
sentiment_model = pipeline("sentiment-analysis", framework="pt")

def hf_sentiment(text):
    if not text.strip():
        return "Neutral"
    result = sentiment_model(text[:512])[0]  # truncate long text
    return result['label'].capitalize()

df["HF_Sentiment"] = df["Cleaned_Message"].apply(hf_sentiment)

# ----------------------------
# Show Sentiment Distribution
# ----------------------------
print("\nVADER Sentiment distribution:")
print(df["VADER_Sentiment"].value_counts())

print("\nHugging Face Sentiment distribution:")
print(df["HF_Sentiment"].value_counts())

# ----------------------------
# Word Cloud
# ----------------------------
all_words = " ".join(df["Cleaned_Message"])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# ----------------------------
# Bar Chart Comparison
# ----------------------------
plt.figure(figsize=(6, 4))
df["HF_Sentiment"].value_counts().plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Hugging Face Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Feedbacks')
plt.xticks(rotation=0)
plt.show()

# ----------------------------
# Save Results
# ----------------------------
output_path = r"C:\Users\User\Downloads\sentiment_resultsHg.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ Results saved to: {output_path}")