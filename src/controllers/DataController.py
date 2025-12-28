from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.helpers.config import get_settings
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)


class TriviaQADataProcessor:

    DATASET_PATH = get_settings().DATASET_PATH
    DATASET_CONFIG = get_settings().DATASET_CONFIG

    def __init__(self,chunk_size: int = 800,chunk_overlap: int = 200,max_samples: int = 500):
        self.max_samples = max_samples

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            is_separator_regex=False
        )

    def load_data(self, split: str = "train") -> List[Dict[str, Any]]:
        logging.info("Loading TriviaQA dataset...")
        dataset = load_dataset(self.DATASET_PATH,self.DATASET_CONFIG,split=split,streaming=True)

        samples = list(dataset.take(self.max_samples))
        logging.info(f"Loaded {len(samples)} samples")
        return samples

    def extract_context(self, example: Dict[str, Any]) -> str | None:
        pages = example.get("entity_pages", {}).get("wiki_context", [])
        if not pages:
            return None

        return pages[0]

    def preprocess(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_documents = []

        for example in raw_data:
            context = self.extract_context(example)
            if not context:
                continue

            chunks = self.text_splitter.split_text(context)

            for chunk in chunks:
                processed_documents.append({
                    "page_content": chunk,
                    "metadata": {
                        "question_id": example.get("question_id"),
                        "answer": example.get("answer", {}).get("value"),
                        "source": "TriviaQA-RC"
                    }
                })

        logging.info(
            f"Generated {len(processed_documents)} document chunks"
        )
        return processed_documents

    def to_langchain_documents(self, processed_data: List[Dict[str, Any]]) -> List[Document]:
        return [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in processed_data
        ]
    
    def process(self, split: str = "train") -> List[Document]:
        raw_data = self.load_data(split=split)
        processed_data = self.preprocess(raw_data)
        langchain_documents = self.to_langchain_documents(processed_data)
        return langchain_documents
    
