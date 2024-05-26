import logging
import time
from typing import Any, List, Optional, Tuple, Type, Union

from numpy import ndarray
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding(
        self,
        sentences: List[str],
        batch_size: int = 32,
        parallel: int = 1,
    ) -> List[List[float]]:
        start = time.time()
        log.info(f"Embedding {len(sentences)} sentences using SentenceTransformer")
        # Embed the sentences
        embeddings = self.model.encode(
            sentences, batch_size=batch_size, convert_to_numpy=True
        )

        # Assert that embeddings is an instance of numpy ndarray
        assert isinstance(
            embeddings, ndarray
        ), "Expected embeddings to be a numpy ndarray"
        elapsed_time=time.time()-start
        return embeddings.tolist(),elapsed_time



class VectorEmbedService:
    def __init__(
        self,
        embedder_type: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.embedder_type = embedder_type
        self.model_name = model_name
        if self.embedder_type == "sentence_transformer":
            self.embedder = SentenceTransformerEmbedder(
                self.model_name
            )  # "all-MiniLM-L6-v2"
        else:
            raise ValueError(f"Embedder type {embedder_type} not recognized.")

    def embed(
        self,
        sentences: List[str],
        batch_size: int = 32,
        parallel: int = 1,
    ) -> List[List[float]]:
        result, elapsed_time = self.embedder.get_embedding(
            sentences, batch_size=batch_size, parallel=parallel
        )
        if elapsed_time != 0.0:
            log.info(
                f"Embedded {len(sentences)} records in {elapsed_time} seconds @ {int(len(sentences)/elapsed_time)} itr/sec."
            )
        return result
