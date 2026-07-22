import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    return [word for word in tokens if word.isalnum()]


preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]

preprocessed_docs

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

query = "machine learning"


def search_tfidf(query, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    sorted_similarities = list(enumerate(similarities))
    results = sorted(sorted_similarities, key=lambda x: x[1], reverse=True)
    return results


search_similarities = search_tfidf(query, vectorizer, tfidf_matrix)

print(f"Top 10 documents similarity score {query}: ")
for doc_index, score in search_similarities[:10]:
    print(f"documento {doc_index}: {documents[doc_index]}")
