# Document search with TF-IDF and cosine similarity.
# Ranks documents by how similar they are to a query.

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    # Normalize text: lowercase, tokenize, keep alphanumeric tokens only.
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    return [word for word in tokens if word.isalnum()]


# Clean each document and join tokens back into a string for the vectorizer.
preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]

# TF-IDF: weights words by how important they are in a document vs the whole corpus.
# fit_transform learns the vocabulary and builds the document-term matrix.
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

query = "machine learning"


def search_tfidf(query, vectorizer, tfidf_matrix):
    # Project the query into the same TF-IDF space as the documents.
    query_vector = vectorizer.transform([query])
    # Cosine similarity: how close the query vector is to each document vector.
    similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    # Pair each document index with its score, then sort highest first.
    sorted_similarities = list(enumerate(similarities))
    results = sorted(sorted_similarities, key=lambda x: x[1], reverse=True)
    return results


search_similarities = search_tfidf(query, vectorizer, tfidf_matrix)

print(f"Top 10 documents similarity score {query}: ")
for doc_index, score in search_similarities[:10]:
    print(f"documento {doc_index}: {documents[doc_index]}")
