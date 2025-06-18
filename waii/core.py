from typing import Union, List, Tuple, Literal
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import os
from huggingface_hub import hf_hub_download
import json
from openai import OpenAI

class TaskExtractor:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        num_tasks: int = 5,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.num_tasks = num_tasks

    def extract(self, description: str) -> List[Tuple[str, float]]:
        system_prompt = (
            "You are an assistant specialized in analyzing job descriptions. "
            f"Your task is to extract up to {self.num_tasks} key tasks, responsibilities, or requirements essential for the role from the provided job description. "
            "Focus only on actionable tasks and skills that are critical to the candidate's performance. Combine similar tasks or requirements into a single task to avoid redundancy.\n"
            "For each task:\n"
            "1. Assign a 'task_num' starting from 1. This number should reflect the order of the task based on its first appearance in the job description.\n"
            "2. Provide a 'task' summary focusing on technical or hard skills.\n"
            "3. Assign a 'weight' (a float up to two decimal places) reflecting the task's importance and relevance within the job description, considering factors such as role focus and key responsibilities. The total should sum to 1.\n"
            "Output the result as a **JSON array** of dictionaries. Each dictionary should contain:\n"
            "- 'task_num': Integer.\n"
            "- 'task': String.\n"
            "- 'weight': Float.\n"
        )

        user_prompt = (
            f"Job Description:\n"
            "\"\"\"\n"
            f"{description}\n"
            "\"\"\"\n"
            "Output:\n"
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        try:
            content = completion.choices[0].message.content.strip()
            parsed = json.loads(content)
            return [(entry["task"], float(entry["weight"])) for entry in parsed]
        except Exception as e:
            raise ValueError(f"Failed to parse model output: {e}\nRaw output:\n{content}")

class PatentRetriever:
    def __init__(self,
                 model_name='nomic-ai/nomic-embed-text-v1.5',
                 base_path=None,
                 device=None,
                 nprobe=10):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

        # Download or load files from HuggingFace hub
        repo_id = "neochoi95/waii_retrieve"
        self.past_index = faiss.read_index(
            hf_hub_download(repo_id=repo_id, filename="faiss_nomic_patent_past_index.ivf")
        )
        self.current_index = faiss.read_index(
            hf_hub_download(repo_id=repo_id, filename="faiss_nomic_patent_current_index.ivf")
        )
        self.past_index.nprobe = nprobe
        self.current_index.nprobe = nprobe

        with open(hf_hub_download(repo_id=repo_id, filename="patent_past_nomic_embed_dict.pkl"), 'rb') as f:
            self.past_dict = pickle.load(f)
        with open(hf_hub_download(repo_id=repo_id, filename="patent_current_nomic_embed_dict.pkl"), 'rb') as f:
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
