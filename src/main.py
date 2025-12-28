import asyncio
from asyncio.log import logger
from fastapi import FastAPI
from src.routers import base
from src.controllers.VectordbControllers import VectorStoreManager
from src.controllers.DataController import TriviaQADataProcessor

app = FastAPI()

app.include_router(base.base_router)

vector_db: VectorStoreManager | None = None

async def init_vector_db():
    global vector_db
    logger.info("Initializing Vector DB in background...")
    vector_db = VectorStoreManager()

    if not vector_db.load_index():
        logger.info("No index found on disk. Building FAISS index...")
        # Reduce samples to 100 for faster initial run as discussed
        processor = TriviaQADataProcessor(max_samples=100)
        
        # Run CPU-intensive processing in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(None, processor.process)
        
        logger.info(f"Processing {len(docs)} document chunks...")
        embeddings = await loop.run_in_executor(None, vector_db.create_embeddings, docs)
        
        await loop.run_in_executor(None, vector_db.build_faiss_index, embeddings, docs)
        await loop.run_in_executor(None, vector_db.save_index)
        logger.info("FAISS index built and saved successfully.")
    else:
        logger.info("FAISS index loaded from disk.")

@app.on_event("startup")
async def startup_event():
    # Start the initialization task without blocking
    asyncio.create_task(init_vector_db())


