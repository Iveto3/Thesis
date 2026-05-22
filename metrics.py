from sentence_transformers import SentenceTransformer, util


def compute_similarity(response: str, reference: str) -> float:
    embeddings = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
        ).encode(
        [response, reference],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return float(score)
