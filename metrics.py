import torch
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rougeLsum"],
    use_stemmer=True,
    split_summaries=True,
)

BERT_SCORER = None


def compute_similarity(response: str, reference: str) -> float:
    """ Computes cosine similarity between output and reference
    output. """
    embeddings = MODEL.encode(
        [response, reference],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    score = util.cos_sim(
        embeddings[0],
        embeddings[1],
    ).item()

    return float(score)


def compute_rouge_lsum(response: str, reference: str) -> float:
    """ Computes ROUGE-Lsum score between output and reference
    output. """
    scores = ROUGE_SCORER.score(
        target=reference,
        prediction=response,
    )

    return float(scores["rougeLsum"].fmeasure)


def get_bert_scorer() -> BERTScorer:
    """ Returns BERTScorer. """
    global BERT_SCORER

    if BERT_SCORER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        BERT_SCORER = BERTScorer(
            model_type="roberta-large",
            lang="en",
            idf=False,
            rescale_with_baseline=False,
            device=device,
        )

    return BERT_SCORER


def compute_bertscore_batch(
    responses: list[str],
    references: list[str],
    batch_size: int = 8,
) -> list[dict[str, float]]:
    """ Computes BERTScore. """
    scorer = get_bert_scorer()

    precision, recall, f1 = scorer.score(
        responses,
        references,
        batch_size=batch_size,
        verbose=False,
    )

    return [
        {
            "bertscore_precision": float(p),
            "bertscore_recall": float(r),
            "bertscore_f1": float(f),
        }
        for p, r, f in zip(
            precision.cpu(),
            recall.cpu(),
            f1.cpu(),
        )
    ]
