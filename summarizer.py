# streamlit_app.py
import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import heapq
import nltk
import nltk
nltk.download('punkt')


# Download NLTK data (one time)
nltk.download('punkt')
nltk.download('stopwords')

st.title("Offline Text Summarizer (No API)")

#  input Text
user_input = st.text_area("Enter your text:")

if st.button("Summarize"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Preprocessing stage
        text = user_input
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text.lower())
        word_frequencies = {}
        for word in words:
            if word not in stop_words and word.isalnum():
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        max_freq = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_freq

        # Sentence scoring
        sentence_list = sent_tokenize(text)
        sentence_scores = {}
        for sent in sentence_list:
            for word in word_tokenize(sent.lower()):
                if word in word_frequencies:
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores:
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        # Get summary ready
        summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)

        st.subheader("Summary:")
        st.write(summary)
