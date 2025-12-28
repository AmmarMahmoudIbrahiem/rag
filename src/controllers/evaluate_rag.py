import time
import json
import logging
from datasets import load_dataset
from src.controllers.VectordbControllers import VectorStoreManager
from src.controllers.LLMController import LLMController
from sentence_transformers import SentenceTransformer, util

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Controllers
vsm = VectorStoreManager()
llm = LLMController()
# Model for calculating Semantic Similarity (Evaluation Metric)
eval_model = SentenceTransformer('all-MiniLM-L6-v2')

def run_evaluation(num_samples: int = 10):
    # 1. Load TriviaQA validation set (streaming to save memory)
    dataset = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
    samples = list(dataset.take(num_samples))
    
    evaluation_results = []
    
    print(f"--- Starting Evaluation on {num_samples} samples ---")

    for i, item in enumerate(samples):
        question = item['question']
        correct_answer = item['answer']['value']
        
        start_time = time.time()
        
        try:
            # Step A: Retrieve Contexts
            contexts = vsm.similarity_search(query=question, k=3)
            
            # Step B: Generate RAG Response
            rag_output = llm.generate_rag_response(question=question, contexts=contexts)
            ai_answer = rag_output['answer']
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Step C: Analyze Accuracy (Semantic Similarity Score)
            emb_ai = eval_model.encode(ai_answer, convert_to_tensor=True)
            emb_gt = eval_model.encode(correct_answer, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb_ai, emb_gt).item()

            evaluation_results.append({
                "sample_id": i + 1,
                "question": question,
                "correct_answer": correct_answer,
                "ai_answer": ai_answer,
                "similarity_score": round(similarity, 4),
                "latency_ms": round(latency_ms, 2),
                "retrieved_count": len(contexts)
            })
            
            print(f"Sample {i+1}: Similarity={round(similarity, 2)} | Latency={round(latency_ms)}ms")

        except Exception as e:
            logger.error(f"Failed sample {i+1}: {e}")

    # 2. Calculate Aggregate Metrics
    avg_accuracy = sum(r['similarity_score'] for r in evaluation_results) / len(evaluation_results)
    avg_latency = sum(r['latency_ms'] for r in evaluation_results) / len(evaluation_results)

    summary = {
        "total_samples": len(evaluation_results),
        "average_similarity": round(avg_accuracy, 4),
        "average_latency_ms": round(avg_latency, 2),
        "detailed_results": evaluation_results
    }

    return summary

if __name__ == "__main__":
    report = run_evaluation(num_samples=20)
    
    # Save results to file
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("\n--- Evaluation Complete ---")
    print(f"Avg Similarity: {report['average_similarity']}")
    print(f"Avg Latency: {report['average_latency_ms']}ms")