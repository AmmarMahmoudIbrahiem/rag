from asyncio.log import logger
from fastapi import FastAPI
from routers import base
from controllers.VectordbControllers import VectorStoreManager
from controllers.DataController import TriviaQADataProcessor

app = FastAPI()

app.include_router(base.base_router)

vector_db: VectorStoreManager | None = None

@app.on_event("startup")
def startup_event():
    global vector_db
    vector_db = VectorStoreManager()

    if not vector_db.load_index():
        logger.info("Building FAISS index at startup...")
        processor = TriviaQADataProcessor()
        docs = processor.process()

        embeddings = vector_db.create_embeddings(docs)
        vector_db.build_faiss_index(embeddings, docs)
        vector_db.save_index()


