import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Fake Job Post Detection", layout="wide")
st.title("Fake Job Post Detection")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Load dataset
if uploaded_file:
    st.success("Custom dataset uploaded successfully!")
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using default sample dataset for demo.")
    df = pd.read_csv("fake_job_postings_sample.csv")  # Default dataset

st.subheader("Dataset Overview")
st.write(df.head())
st.write(f"**Total Rows:** {len(df)}")
st.write(df['fraudulent'].value_counts())

# Combine text columns into one
df['text'] = df[['title', 'location', 'company_profile', 'description',
                 'requirements', 'benefits']].fillna('').agg(' '.join, axis=1)

X = df['text']
y = df['fraudulent']

# TF-IDF Vectorization with bigrams
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model with class weighting
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Predict with custom threshold
y_prob = model.predict_proba(X_test)[:, 1]
custom_threshold = 0.4  # lower threshold = more sensitive to fake jobs
y_pred = (y_prob >= custom_threshold).astype(int)

# Show model performance
st.subheader("Model Performance")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.text(classification_report(y_test, y_pred))

# User input for custom prediction
st.subheader("Test Your Own Job Description")
job_post = st.text_area("Enter Job Description Here...")

if st.button("Predict"):
    if job_post.strip():
        job_vec = vectorizer.transform([job_post])
        job_prob = model.predict_proba(job_vec)[:, 1][0]
        prediction = 1 if job_prob >= custom_threshold else 0
        result = "**Fake Job Posting!**" if prediction == 1 else "**Real Job Posting!**"
        st.markdown(f"### Prediction: {result}")
    else:
        st.warning("Please enter a job description to predict.")

