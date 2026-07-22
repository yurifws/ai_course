# Tokenization basics with NLTK.
# Splits text into words/sentences and shows a simple preprocess step.

import nltk

# Download the tokenizer models used by word_tokenize / sent_tokenize.
nltk.download("punkt_tab")

text = "Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. Instead of following fixed rules, a model is trained on examples and gradually improves its predictions as it sees more information. Common tasks include classification, regression, clustering, and natural language processing. Popular approaches range from simple linear models to deep neural networks with millions of parameters. As data grows and computing power increases, machine learning continues to power applications such as recommendation systems, image recognition, fraud detection, and virtual assistants."

# Word tokenization: breaks the text into individual tokens (words and punctuation).
word_tokens = nltk.word_tokenize(text)
print(word_tokens)

# Sentence tokenization: splits the text into a list of sentences.
sentence_tokens = nltk.sent_tokenize(text)
print(sentence_tokens)


def preprocess(text):
    # Lowercase + tokenize, then keep only alphanumeric tokens (drop punctuation).
    tokens = nltk.word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum()]


# Small corpus used to practice preprocessing on multiple documents.
documents = [
    "Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed.",
    "Instead of following fixed rules, a model is trained on examples and gradually improves its predictions as it sees more information.",
    "Common tasks include classification, regression, clustering, and natural language processing.",
    "Popular approaches range from simple linear models to deep neural networks with millions of parameters.",
    "As data grows and computing power increases, machine learning continues to power applications such as recommendation systems, image recognition, fraud detection, and virtual assistants.",
]

# Rebuild each document as a cleaned string of tokens joined by spaces.
preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]

print(preprocessed_docs)
