import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set Streamlit app title
st.title("📰 Fake News Detection App")

# Sidebar options
st.sidebar.header("Options")
choice = st.sidebar.selectbox("Choose Action", ["Train Model", "Test News Article"])

# Initialize session state variables
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "model" not in st.session_state:
    st.session_state.model = None

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower().strip()

if choice == "Train Model":
    st.header("📂 Upload Dataset for Training")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        # Load dataset
        data = pd.read_csv(uploaded_file, on_bad_lines='skip')

        st.write("Dataset preview:")
        st.dataframe(data.head())

        # Check if required columns exist
        required_columns = {"text", "label"}
        data.columns = map(str.lower, data.columns)  # Convert column names to lowercase
        if required_columns.issubset(set(data.columns)):
            # Ensure label is numerical
            if data["label"].dtype == object:
                data["label"] = data["label"].map({"Real": 1, "Fake": 0})

            # Clean text data
            data["text"] = data["text"].astype(str).apply(clean_text)

            # Drop missing values
            data = data[["text", "label"]].dropna()
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                data["text"], data["label"], test_size=0.2, random_state=42
            )

            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_df=0.7,
                max_features=10000,
                ngram_range=(1, 2)
            )
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            # Train Random Forest classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_tfidf, y_train)

            # Save trained model in session state
            st.session_state.vectorizer = vectorizer
            st.session_state.model = model

            # Evaluate model
            y_pred = model.predict(X_test_tfidf)
            st.success("🎉 Model Training Complete!")
            st.write(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.write("📊 Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("📉 Confusion Matrix:")
            st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))
        else:
            st.error("❌ The dataset must contain 'Text' and 'label' columns.")

elif choice == "Test News Article":
    st.header("📝 Test News Article")

    # Check if model is trained
    if st.session_state.vectorizer is None or st.session_state.model is None:
        st.warning("⚠️ Please train the model first by uploading a dataset.")
    else:
        # Text input for news article
        news_article = st.text_area("Enter the news article content:")
        
        if st.button("Classify"):
            if news_article.strip():
                # Preprocess and predict
                cleaned_article = clean_text(news_article)
                input_tfidf = st.session_state.vectorizer.transform([cleaned_article])
                prediction = st.session_state.model.predict(input_tfidf)
                result = "✅ Real" if prediction[0] == 1 else "❌ Fake"
                st.subheader(f"The article is classified as: {result}")
            else:
                st.warning("⚠️ Please enter some text to classify.")
