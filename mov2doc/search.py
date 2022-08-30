from sentence_transformers import SentenceTransformer, util
import torch
from nltk.tokenize import sent_tokenize


device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)


def sementic_search(corpus, query, k=1):
    dt = sent_tokenize(corpus)
    corpus_embeddings = embedder.encode(dt, convert_to_tensor=True)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(dt))
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=k)
    return "\n".join(dt[i] for i in top_results[1])
