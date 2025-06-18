from typing import Union, List, Tuple, Literal
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import os

class PatentRetriever:
    def __init__(self,
                 model_name='nomic-ai/nomic-embed-text-v1.5',
                 base_path=None,
                 device=None,
                 nprobe=10):
        base_dir = base_path or os.path.join(os.path.dirname(__file__), 'patent_embeddings')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

        # Load full (unsplit) files directly
        self.past_index = faiss.read_index(os.path.join(base_dir, 'faiss_nomic_patent_past_index.ivf'))
        self.current_index = faiss.read_index(os.path.join(base_dir, 'faiss_nomic_patent_current_index.ivf'))

        self.past_index.nprobe = nprobe
        self.current_index.nprobe = nprobe

        with open(os.path.join(base_dir, 'patent_past_nomic_embed_dict.pkl'), 'rb') as f:
            self.past_dict = pickle.load(f)
        with open(os.path.join(base_dir, 'patent_current_nomic_embed_dict.pkl'), 'rb') as f:
            self.current_dict = pickle.load(f)

    def _get_embedding(self, text: str) -> np.ndarray:
        emb = self.model.encode("search_document: " + text)
        return np.array(emb, dtype='float32')

    def _similarity(self, query_embedding: np.ndarray, target_embedding: np.ndarray) -> float:
        return cosine_similarity(query_embedding.reshape(1, -1),
                                 np.array(target_embedding).reshape(1, -1))[0][0]

    def _retrieve_best_match(self, embedding: np.ndarray, index, embed_dict) -> float:
        D, I = index.search(embedding.reshape(1, -1), 1)
        best_id = I[0][0]
        return self._similarity(embedding, embed_dict[best_id]['Embedding'])

    def ai_exposure(
        self,
        inputs: Union[str, List[str], List[Tuple[str, float]]],
        weight: Union[Literal['softmax'], None] = None
    ) -> dict:
        if isinstance(inputs, str):
            tasks = [inputs]
        elif isinstance(inputs, list) and all(isinstance(i, str) for i in inputs):
            tasks = inputs
        elif isinstance(inputs, list) and all(isinstance(i, tuple) and isinstance(i[1], float) for i in inputs):
            tasks = [i[0] for i in inputs]
        else:
            raise ValueError("Invalid input format.")

        embeddings = [self._get_embedding(task) for task in tasks]

        if weight == 'softmax':
            weights = softmax(np.mean(np.vstack(embeddings), axis=1))
        elif isinstance(inputs, list) and all(isinstance(i, tuple) and isinstance(i[1], float) for i in inputs):
            weights = [i[1] for i in inputs]
        else:
            weights = [1.0 / len(tasks)] * len(tasks)

        past_scores = [w * self._retrieve_best_match(e, self.past_index, self.past_dict)
                       for e, w in zip(embeddings, weights)]
        current_scores = [w * self._retrieve_best_match(e, self.current_index, self.current_dict)
                          for e, w in zip(embeddings, weights)]

        best_past_ids = [self.past_index.search(e.reshape(1, -1), 1)[1][0][0] for e in embeddings]
        best_current_ids = [self.current_index.search(e.reshape(1, -1), 1)[1][0][0] for e in embeddings]

        return {
            "past_ai_exposure": sum(past_scores),
            "current_ai_exposure": sum(current_scores),
            "past_patent_abstracts": [self.past_dict[i]['Abstract'] for i in best_past_ids],
            "current_patent_abstracts": [self.current_dict[i]['Abstract'] for i in best_current_ids],
        }
