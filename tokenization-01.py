import nltk

nltk.download("punkt_tab")

text = "Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. Instead of following fixed rules, a model is trained on examples and gradually improves its predictions as it sees more information. Common tasks include classification, regression, clustering, and natural language processing. Popular approaches range from simple linear models to deep neural networks with millions of parameters. As data grows and computing power increases, machine learning continues to power applications such as recommendation systems, image recognition, fraud detection, and virtual assistants."

word_tokens = nltk.word_tokenize(text)
print(word_tokens)

sentence_tokens = nltk.sent_tokenize(text)
print(sentence_tokens)


def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum()]


documents = [
    "Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed.",
    "Instead of following fixed rules, a model is trained on examples and gradually improves its predictions as it sees more information.",
    "Common tasks include classification, regression, clustering, and natural language processing.",
    "Popular approaches range from simple linear models to deep neural networks with millions of parameters.",
    "As data grows and computing power increases, machine learning continues to power applications such as recommendation systems, image recognition, fraud detection, and virtual assistants.",
]

preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]

print(preprocessed_docs)
