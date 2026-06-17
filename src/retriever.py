from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from src import config

class ThresholdRetriever(BaseRetriever):
    """
    A custom retriever that wraps a FAISS vector store and filters out
    retrieved documents that do not meet the minimum similarity threshold.
    """
    vector_store: Any
    k: int = config.RETRIEVAL_K
    score_threshold: float = config.SIMILARITY_SCORE_THRESHOLD

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # Use similarity_search_with_score which returns raw cosine scores in [0, 1]
        # when DistanceStrategy.COSINE is used with normalized embeddings
        results = self.vector_store.similarity_search_with_score(
            query,
            k=self.k
        )

        # Filter documents based on score threshold
        relevant_docs = []
        for doc, score in results:
            float_score = float(score)
            if float_score >= self.score_threshold:
                # Store the score in metadata for reference/citation/debugging
                doc.metadata["similarity_score"] = float_score
                relevant_docs.append(doc)

        return relevant_docs
