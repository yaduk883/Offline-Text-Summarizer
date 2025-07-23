import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import heapq

# 🔽 Download necessary NLTK resources (this uses internet)
nltk.download('punkt')
nltk.download('stopwords')

# 🔤 Input text
text = """
The Rashtrapati Bhavan houses the magnificent third century B.C. sandstone capital of the Ashokan Pillar known as the Rampurva Bull.
It is one of the finest examples of Mauryan art.
This capital was discovered in the 19th century at Rampurva in Bihar.
It is now displayed prominently at the entrance hall of Rashtrapati Bhavan as a symbol of India’s rich heritage.
"""

# 🔣 Tokenize text into sentences and words
sentences = sent_tokenize(text)
words = word_tokenize(text.lower())

# 🛑 Remove stopwords and punctuation
stop_words = set(stopwords.words("english"))
words = [word for word in words if word.isalnum() and word not in stop_words]

# 📊 Frequency distribution
freq_dist = FreqDist(words)

# 🧠 Score sentences based on word frequency
sentence_scores = {}
for sentence in sentences:
    for word in word_tokenize(sentence.lower()):
        if word in freq_dist:
            if sentence not in sentence_scores:
                sentence_scores[sentence] = freq_dist[word]
            else:
                sentence_scores[sentence] += freq_dist[word]

# ✂️ Summarize (top 2 sentences)
summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
summary = " ".join(summary_sentences)

# 📤 Output
print("\n📄 Original Text:\n", text)
print("\n📝 Summary:\n", summary)
