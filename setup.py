
from setuptools import setup, find_packages

setup(
    name="waii",
    version="0.1.0",
    description="Weighted AI Exposure Index (wAII) tool using FAISS and SentenceTransformer",
    author="Eun Cheol Choi",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "waii": ["patent_embeddings/*"]
    },
    install_requires=[
        "torch",
        "sentence-transformers",
        "faiss-cpu",
        "scikit-learn",
        "numpy"
    ],
)
