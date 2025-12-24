import cohere
import logging
import time
from typing import Dict, List, Any, Optional
from helpers.config import get_settings
import backoff

logger = logging.getLogger(__name__)

# Normalize Cohere exception class
try:
    CohereAPIError = cohere.CohereAPIError
except Exception:
    CohereAPIError = getattr(cohere, "CohereError", Exception)

class LLMController:
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self._init_client()
    
    def _init_client(self):
        if not self.settings.COHERE_API_KEY or self.settings.COHERE_API_KEY == "your_cohere_api_key_here":
            logger.warning("Cohere API key not configured properly")
            return
        
        try:
            self.client = cohere.Client(self.settings.COHERE_API_KEY)
            # Recommendation: Use 'command-r' or 'command-r-plus' for RAG
            logger.info(f"Cohere client initialized for Chat API with model: {self.settings.COHERE_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            raise

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=30
    )
    def generate_answer(self, question: str, contexts: List[Dict[str, Any]],
                        max_tokens: int = 500,
                        temperature: float = 0
                        ) -> Dict[str, Any]:
        if not self.client:
            raise ValueError("Cohere client not initialized. Please check your API key.")
        
        if not contexts:
            return {
                "answer": "I cannot answer this question because no relevant context was found.",
                "metadata": {"model": self.settings.COHERE_MODEL, "error": "no_contexts"}
            }
        
        # 1. Prepare the context block
        context_block = "\n".join([f"Context {i+1}: {ctx['content']}" for i, ctx in enumerate(contexts)])
        
        # 2. Construct the Chat message
        # We prompt the model to act as a grounded assistant
        chat_message = f"""Use the following contexts to provide a concise answer to the question. 
If the answer isn't in the context, state that you don't know.

Contexts:
{context_block}

Question: {question}"""
        
        start_time = time.time()
        
        try:
            # 3. Use the .chat() endpoint (Legacy .generate is removed)
            response = self.client.chat(
                model=self.settings.COHERE_MODEL,
                message=chat_message,
                temperature=temperature or self.settings.COHERE_TEMPERATURE,
                # max_tokens in Chat API limits the response length
                max_tokens=max_tokens or self.settings.COHERE_MAX_TOKENS
            )

            generation_time = (time.time() - start_time) * 1000
            
            return {
                "answer": response.text.strip(),
                "metadata": {
                    "model": self.settings.COHERE_MODEL,
                    "generation_time_ms": round(generation_time, 2),
                    "finish_reason": getattr(response, 'finish_reason', 'COMPLETE')
                }
            }

        except Exception as e:
            logger.error(f"Error in Cohere Chat API: {e}")
            raise

    def generate_rag_response(self, question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Generate the text answer using the new Chat logic
        llm_result = self.generate_answer(question, contexts)
        
        # Format context for the final API response
        formatted_contexts = [
            {
                "content": ctx["content"],
                "metadata": ctx.get("metadata", {}),
                "score": ctx.get("score", 0.0),
                "rank": ctx.get("rank", i + 1)
            }
            for i, ctx in enumerate(contexts)
        ]
        
        return {
            "question": question,
            "answer": llm_result["answer"],
            "contexts": formatted_contexts,
            "llm_metadata": llm_result["metadata"]
        }