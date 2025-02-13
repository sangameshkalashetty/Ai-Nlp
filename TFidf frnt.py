import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Streamlit App Title
st.title("TF-IDF Vectorizer App")

# User input for text
user_input = st.text_area("Enter text:", "Type here...")

# Process input
if st.button("Calculate TF-IDF"):
    if user_input.strip():
        corpus = [user_input]  # Convert input to list
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Display results
        st.subheader("TF-IDF Features:")
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        for word, score in zip(feature_names, tfidf_scores):
            st.write(f"**{word}**: {score:.4f}")
    else:
        st.warning("Please enter some text.")

