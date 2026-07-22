# Document ranking with BM25 (Okapi BM25).
# Scores how well each document matches a query using term frequency and rarity.

import nltk
import numpy as np
from rank_bm25 import BM25Okapi


# Corpus of short documents about machine learning.
documents = [
    "Machine learning is a field of artificial intelligence that allows computers to learn patterns from data.",
    "Machine learning gives systems the ability to improve their performance without being explicitly programmed.",
    "Instead of following only fixed rules, machine learning discovers hidden relationships in data.",
    "This field combines statistics, algorithms, and computing power to extract knowledge.",
    "The goal is to create models capable of generalizing beyond the examples seen during training.",
    "Machine learning applications range from movie recommendations to medical diagnoses.",
    "Machine learning algorithms transform raw data into useful predictions.",
    "Unlike traditional software, ML adapts as new data arrives.",
    "Learning can be supervised, unsupervised, or reinforcement, depending on the type of problem.",
    "In practice, machine learning is the engine that drives many advances in computer vision and natural language processing.",
    "More than finding patterns, machine learning helps make decisions based on evidence.",
]


def preprocess(text):
    # Lowercase + tokenize; keep alphanumeric tokens (punctuation removed).
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    return [word for word in tokens if word.isalnum()]


# BM25 expects a list of token lists (one per document).
tokenized_docs = [preprocess(doc) for doc in documents]

# Do not remove stopwords for BM25: the algorithm uses IDF and can use them.
bm25 = BM25Okapi(tokenized_docs)

query = "machine learning"


def search_bm25(query, bm25):
    # Tokenize the query the same way as the documents, then get a score per doc.
    tokenized_query = preprocess(query)
    results = bm25.get_scores(tokenized_query)
    return results


results = search_bm25(query, bm25)

# Print documents from highest BM25 score to lowest.
for i in np.argsort(results)[::-1]:
    print(f"document {i}: {documents[i]}")
