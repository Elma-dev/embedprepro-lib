from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .embedding_service import VectorEmbedService


class AgglomerativeClustering:
    def __init__(self, embedding_service: Optional[VectorEmbedService] = None):
        """
        Initializes the AgglomerativeClustering class with an optional external embedding service.

        Parameters:
            embedding_service (VectorEmbedService, optional): An external service for embedding sentences.
        """
        self.embedding_service = embedding_service

    def _embed_sentences(
        self, sentences: List[str], batch_size: int = 32, parallel: bool = False
    ) -> np.ndarray:
        """
        Embeds the given sentences using the internal or external embedding service.

        Parameters:
            sentences (List[str]): A list of sentences to embed.
            batch_size (int, optional): The batch size to use for embedding. Defaults to 32.
            parallel (bool, optional): Flag to indicate if embedding should be done in parallel. Defaults to False.

        Returns:
            np.ndarray: An array of sentence embeddings.
        """
        if self.embedding_service is None:
            raise ValueError("No embedding service provided.")
        return np.array(
            self.embedding_service.embed(
                sentences, batch_size=batch_size, parallel=parallel
            )
        )

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Normalizes the given embeddings.

        Parameters:
            embeddings (np.ndarray): An array of sentence embeddings.

        Returns:
            np.ndarray: An array of normalized sentence embeddings.
        """
        return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    def cluster(
        self,
        sentences: List[str],
        threshold: float,
        min_cluster_size: int,
        batch_size: int = 32,
        parallel: bool = False,
        show_progress_bar: bool = True,
    ) -> List[List[int]]:
        """
        Clusters the given sentences into n_clusters using agglomerative clustering.

        Parameters:
            sentences (List[str]): A list of sentences to cluster.
            threshold (float): The threshold for considering sentences as redundant.
            min_cluster_size (int): The minimum size of a cluster.
            batch_size (int, optional): The batch size to use for embedding. Defaults to 32.
            parallel (bool, optional): Flag to indicate if embedding should be done in parallel. Defaults to False.
            show_progress_bar (bool, optional): Flag to indicate if a progress bar should be shown. Defaults to False.

        Returns:
            List[List[int]]: A list of clusters, where each cluster is a list of indices corresponding to sentences.
        """
        embeddings = self._embed_sentences(
            sentences, batch_size=batch_size, parallel=parallel
        )
        embeddings = self._normalize_embeddings(embeddings)
        return self._find_clusters(
            embeddings, threshold, min_cluster_size, batch_size, show_progress_bar
        )

    def cluster_with_embeddings(
        self,
        embeddings: np.ndarray,
        threshold: float,
        min_cluster_size: int,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> List[List[int]]:
        """
        Clusters the given sentences into n_clusters using agglomerative clustering.

        Parameters:
            embeddings (np.ndarray): An array of sentence embeddings.
            threshold (float): The threshold for considering sentences as redundant.
            min_cluster_size (int): The minimum size of a community.
            batch_size (int, optional): The batch size to use for embedding. Defaults to 32.
            parallel (bool, optional): Flag to indicate if embedding should be done in parallel. Defaults to False.
            show_progress_bar (bool, optional): Flag to indicate if a progress bar should be shown. Defaults to False.

        Returns:
            List[List[int]]: A list of clusters, where each cluster is a list of indices corresponding to sentences.
        """
        return self._find_clusters(
            embeddings, threshold, min_cluster_size, batch_size, show_progress_bar
        )

    def _find_clusters(
        self,
        embeddings: np.ndarray,
        threshold: float,
        min_cluster_size: int,
        batch_size: int,
        show_progress_bar: bool,
    ) -> List[List[int]]:
        """
        Clusters the given embeddings into communities using agglomerative clustering.

        Parameters:
            embeddings (np.ndarray): An array of sentence embeddings.
            threshold (float): The threshold for considering sentences as redundant.
            min_cluster_size (int): The minimum size of a community.
            batch_size (int): The batch size to use for clustering.
            show_progress_bar (bool): Flag to indicate if a progress bar should be shown.

        Returns:
            List[List[int]]: A list of communities, where each community is a list of indices of sentences.
        """

        # Form initial clusters
        initial_clusters = self._form_initial_clusters(
            embeddings, threshold, min_cluster_size, batch_size, show_progress_bar
        )

        # Filter out overlapping clusters
        unique_clusters = self._filter_overlapping_clusters(
            initial_clusters, min_cluster_size
        )

        return unique_clusters

    @staticmethod
    def _form_initial_clusters(
        embeddings: np.ndarray,
        threshold: float,
        min_cluster_size: int,
        batch_size: int,
        show_progress_bar: bool,
    ) -> List[List[int]]:
        """
        Forms initial clusters from the given embeddings using agglomerative clustering.

        Parameters:
            embeddings (np.ndarray): An array of sentence embeddings.
            threshold (float): The threshold for considering sentences as redundant.
            min_cluster_size (int): The minimum size of a community.
            batch_size (int): The batch size to use for clustering.
            show_progress_bar (bool): Flag to indicate if a progress bar should be shown.

        Returns:
            List[List[int]]: A list of initial clusters, where each cluster is a list of indices of sentences.
        """
        extracted_clusters = []
        for start_idx in tqdm(
            range(0, len(embeddings), batch_size),
            desc="Forming initial clusters",
            disable=not show_progress_bar,
        ):
            # Compute cosine similarity for the current batch against all embeddings
            cos_scores = np.dot(
                embeddings[start_idx : start_idx + batch_size], embeddings.T
            )

            for i in range(cos_scores.shape[0]):
                # Identifying indices where cosine similarity is above the threshold
                above_threshold_indices = np.where(cos_scores[i] >= threshold)[0]

                # Filter clusters based on size
                if len(above_threshold_indices) >= min_cluster_size:
                    # Sort indices based on cosine similarity
                    sorted_indices = above_threshold_indices[
                        np.argsort(cos_scores[i, above_threshold_indices])[::-1]
                    ]
                    extracted_clusters.append(sorted_indices.tolist())

        return sorted(extracted_clusters, key=lambda x: len(x), reverse=True)

    @staticmethod
    def _filter_overlapping_clusters(
        clusters: List[List[int]], min_cluster_size: int
    ) -> List[List[int]]:
        """
        Filters out overlapping clusters to ensure uniqueness.

        Parameters:
            clusters (List[List[int]]): Initial clusters to filter.
            min_cluster_size (int): Minimum size of a cluster to be retained.

        Returns:
            List[List[int]]: Unique clusters after filtering.
        """
        unique_clusters = []
        extracted_ids = set()
        for cluster in clusters:
            non_overlapped_cluster = [
                idx for idx in cluster if idx not in extracted_ids
            ]

            if len(non_overlapped_cluster) >= min_cluster_size:
                unique_clusters.append(non_overlapped_cluster)
                extracted_ids.update(non_overlapped_cluster)

        return sorted(unique_clusters, key=lambda x: len(x), reverse=True)
