import requests
import json

# Test data
test_data = {
    "query": "What is the capital of China?",
    "documents": [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "Paris is the capital of France.",
        "Beijing is a major city in China with many historical sites."
    ],
    "instruction": "Given a web search query, retrieve relevant passages that answer the query"
}

def test_rerank_api():
    """Test the rerank API endpoint"""
    try:
        # Test rerank endpoint
        response = requests.post("http://localhost:8000/rerank", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("Rerank API Test - Success!")
            print(f"Query: {test_data['query']}")
            print("\nRanked Results:")
            for i, result_item in enumerate(result['results']):
                print(f"{i+1}. Score: {result_item['score']:.4f} | Document: {result_item['document']}")
        else:
            print(f"Rerank API Test - Failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Connection error - Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error during testing: {e}")

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            result = response.json()
            print("Health Check - Success!")
            print(f"Status: {result['status']}")
            print(f"Embedding Model: {result['embedding_model']}")
            print(f"Reranker Model: {result['reranker_model']}")
            print(f"Device: {result['device']}")
        else:
            print(f"Health Check - Failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Health check error: {e}")

if __name__ == "__main__":
    print("Testing Qwen3 Text Service...")
    print("="*50)
    
    # Test health check first
    test_health_check()
    print()
    
    # Test rerank API
    test_rerank_api()
