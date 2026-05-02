from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_similarity(response: str, summary: str) -> float:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([response, summary])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
