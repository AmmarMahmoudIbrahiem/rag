import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from controllers import TriviaQADataProcessor
from helpers.config import get_settings

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorStoreManager:
    def __init__(self):
       
        self.settings = get_settings()
        
        self.embedding_model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []

        self.index_path = Path("vector_index.faiss")
        self.metadata_path = Path("metadata.pkl")

        self._init_embedding_model()

    def _init_embedding_model(self):
        """Initialize the SentenceTransformer model."""
        logger.info(f"Loading embedding model: {self.settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            self.settings.EMBEDDING_MODEL,
            device=self.settings.EMBEDDING_DEVICE
        )

    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Convert document page content into numerical vectors."""
        texts = [doc.page_content for doc in documents]
        logger.info(f"Creating embeddings for {len(texts)} documents")

        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings.astype("float32")
    

    def build_faiss_index(self, embeddings: np.ndarray, documents: List[Document]):
        """Build and populate the FAISS index."""
        dimension = embeddings.shape[1]

        # Select metric based on config
        if self.settings.SIMILARITY_METRIC == "cosine":
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(dimension) # Inner Product for Cosine
        else:
            self.index = faiss.IndexFlatL2(dimension) # Euclidean Distance

        self.index.add(embeddings)

        # Map vectors to their original text and metadata
        self.metadata = [
            {
                "doc_id": i,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for i, doc in enumerate(documents)
        ]
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors.")



    def save_index(self):
        """Persist index and metadata to disk."""
        if self.index is None:
            logger.error("No index found to save.")
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Successfully saved index to {self.index_path}")

    def load_index(self) -> bool:
        """Load index and metadata from disk."""
        if not (self.index_path.exists() and self.metadata_path.exists()):
            return False

        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors.")
        return True

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query."""
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        query_embedding = self.embedding_model.encode(
            [query],
            show_progress_bar=False
        ).astype("float32")

        if self.settings.SIMILARITY_METRIC == "cosine":
            faiss.normalize_L2(query_embedding)

        return query_embedding

    def similarity_search(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        if self.index is None:
            logger.info("Index is None, attempting to load from disk...")
            if not self.load_index():
                raise ValueError("Index not found. Please build the index first.")

        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty")

        if k is None:
            k = getattr(self.settings, "TOP_K", 5)

        logger.info(f"Performing similarity search for query: '{query[:50]}...' with k={k}")

        query_embedding = self.embed_query(query)

        distances, indices = self.index.search(
            query_embedding,
            min(k, self.index.ntotal)
        )

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue

            results.append({
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "content": self.metadata[idx]["page_content"],
                "metadata": self.metadata[idx]["metadata"],
            })

        logger.info(f"Returning {len(results)} results")
        return results
    
    
    def build_index_from_dataset(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Main entry point for similarity search. 
        Handles lazy loading and automated building of the FAISS index.
        """
        # Step 1: Ensure Index is Ready
        if self.index is None:
            logger.info("Index not in memory. Checking disk...")
            
            # Try to load from disk first
            if not self.load_index():
                logger.info("No index found on disk. Starting build process from dataset...")
                
                # Initialize Data Processor
                processor = TriviaQADataProcessor()
                documents = processor.process()
                
                if not documents:
                    logger.error("Data processor returned empty document list.")
                    return []

                # Create Embeddings and Build Index
                embeddings = self.create_embeddings(documents)
                self.build_faiss_index(embeddings, documents)
     
                self.save_index()
            else:
                logger.info("Index successfully loaded from disk.")

        try:
            results = self.similarity_search(query=query, k=k)
            return results
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []
    

    

