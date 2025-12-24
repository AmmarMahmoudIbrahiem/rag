import logging
from fastapi import APIRouter
from helpers.config import get_settings
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from controllers.DataController import TriviaQADataProcessor
from controllers.VectordbControllers import VectorStoreManager
from controllers.LLMController import LLMController


base_router = APIRouter()

vector_db = VectorStoreManager()
llm_controller = LLMController()

@base_router.get("/welcome")
def welcome():
    settings = get_settings()
    return { 
        "app_name": settings.app_name, 
        "app_version": settings.app_version
        }


@base_router.get("/process_data")
async def process_data():
    processor = TriviaQADataProcessor()
    chunks = processor.process(split="train")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"Processed {len(chunks)} document chunks from TriviaQA dataset."
        }
    )

class SearchRequest(BaseModel):
    query: str
    k: int = 5  # Number of top similar documents to retrieve
@base_router.post("/search")
def search(req: SearchRequest):
    try:
        return vector_db.similarity_search(req.query, req.k)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logging.exception("Unexpected error during search")
        raise HTTPException(status_code=500, detail="Internal server error")
    



@base_router.post("/ask-rag")
def ask_question(query: str):
    # Step 1: Instant retrieval from your 24,233 chunks
    search_results = vector_db.similarity_search(query=query, k=3)
    
    # Step 2: Pass search_results directly to the LLMController
    if not search_results:
        return {"answer": "No relevant documents found in the TriviaQA database."}
        
    final_response = llm_controller.generate_rag_response(question=query, contexts=search_results)
    
    return final_response
