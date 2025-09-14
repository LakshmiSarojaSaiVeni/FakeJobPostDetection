import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------------------------
# Function to Plot WordCloud
# -------------------------------------------------
def plot_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color="white",
                   stopwords=STOPWORDS, max_words=100).generate(" ".join(text))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16)
    st.pyplot(fig)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Fake Job Post Detector", layout="wide")
st.title("Fake Job Post Detection with Machine Learning")
st.markdown("Upload a dataset, train a model, visualize patterns, and test custom job descriptions.")

# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload Fake Job Postings CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset Loaded! Shape: {df.shape}")
    st.dataframe(df.head())

    # Drop rows without job description
    df = df.dropna(subset=['description'])
    X = df['description']
    y = df['fraudulent']

    # Show word clouds
    st.subheader("Word Cloud Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        plot_wordcloud(df[df['fraudulent'] == 1]['description'], "Fake Job Postings")
    with col2:
        plot_wordcloud(df[df['fraudulent'] == 0]['description'], "Real Job Postings")

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red", fontsize=14)
    st.pyplot(fig)

    # Top words visualization
    st.subheader("Top Predictive Words")
    feature_names = vectorizer.get_feature_names_out()
    sorted_idx = model.coef_[0].argsort()
    top_fake_words = feature_names[sorted_idx[-15:]]
    top_real_words = feature_names[sorted_idx[:15]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_fake_words, model.coef_[0][sorted_idx[-15:]], color="red")
    ax.set_title("Top Words Indicating FAKE Jobs")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_real_words, model.coef_[0][sorted_idx[:15]], color="green")
    ax.set_title("Top Words Indicating REAL Jobs")
    st.pyplot(fig)

    # Custom Job Prediction
    st.subheader("Test Your Own Job Description")
    user_input = st.text_area("Enter Job Description Here...")
    if st.button("Predict"):
        if user_input.strip():
            test_vec = vectorizer.transform([user_input])
            pred = model.predict(test_vec)
            result = "**FAKE Job Posting!**" if pred[0] == 1 else "**REAL Job Posting!**"
            st.markdown(result)
        else:
            st.warning("Please enter a job description.")
else:
    st.info("Please upload the Fake Job Postings CSV file to start.")
