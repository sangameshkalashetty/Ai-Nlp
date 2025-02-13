import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Title
st.title("Bag of Words (BOW) NLP App")

# Text input
user_input = st.text_area("Enter your text here:")

# Process the input when the user submits
if st.button("Generate BOW"):
    if user_input:
        # Convert input text into a list
        text_list = [user_input]

        # Create BOW model
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(text_list)

        # Convert to DataFrame for better visualization
        bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Show result
        st.write("Bag of Words Representation:")
        st.dataframe(bow_df)
    else:
        st.warning("Please enter some text first.")
