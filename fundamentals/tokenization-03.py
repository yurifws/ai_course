# Boolean search over documents with Whoosh (an inverted index library).
# Builds a local index and runs a query with AND logic.

import os
import shutil
import nltk
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser

import warnings

# Stopwords list used by the preprocess helper below.
nltk.download("stopwords")

warnings.filterwarnings("ignore", category=SyntaxWarning)

# Corpus indexed and searched later.
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
    # Lowercase, tokenize, drop punctuation, then remove English stopwords.
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    tokens = [word for word in tokens if word.isalnum()]
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokens = [word for word in tokens if word not in stopwords]
    return tokens


# Quick demo of preprocessing on a single sentence (result is not used further).
text = "Machine learning is a field of artificial intelligence that allows computers to learn patterns from data."
preprocess(text)

# Recreate the index folder so each run starts from a clean index.
if os.path.exists("index_dir"):
    shutil.rmtree("index_dir")
os.mkdir("index_dir")

# Schema: title is a unique ID; content is the searchable text we store.
schema = Schema(title=ID(stored=True, unique=True), content=TEXT(stored=True))

index = create_in("index_dir", schema)

# Write every document into the index, then commit to disk.
writter = index.writer()
for i, doc in enumerate(documents):
    writter.add_document(title=str(i), content=doc)
writter.commit()

# Boolean query: both terms must appear in a matching document.
query = "machine AND learning"


def boolean_search(query, index):
    # Parse the query string against the "content" field.
    parser = QueryParser("content", schema=index.schema)
    parsed_query = parser.parse(query)

    # Search and return (title, content) for each hit.
    with index.searcher() as searcher:
        results = searcher.search(parsed_query)
        return [(hit["title"], hit["content"]) for hit in results]


boolean_search(query, index)
