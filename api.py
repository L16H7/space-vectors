from flask import Flask, request, jsonify
from flask_cors import CORS
import weaviate
import torch
from transformers import AutoModel, AutoProcessor
import argparse
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
import os

app = Flask(__name__)
CORS(app)

# Global variables
model = None
processor = None
weaviate_client = None
collection_name = None

def load_model():
    """Load the SigLIP model and processor if not already loaded."""
    global model, processor
    if model is None or processor is None:
        print("Loading SigLIP model and processor...")
        model = AutoModel.from_pretrained("google/siglip-base-patch16-384")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
        print("Model loaded successfully")
    return model, processor

def generate_text_embedding(text):
    """Generate embedding for the given text query."""
    model, processor = load_model()
    
    # Process the text input
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    
    # Return as numpy array
    return outputs[0].cpu().numpy()

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests with text queries."""
    global weaviate_client, collection_name
    
    # Get the JSON data from the request
    data = request.json
    if not data or 'context' not in data:
        return jsonify({'error': 'Missing context parameter'}), 400
    
    search_term = data['context']
    limit = data.get('limit', 10)  # Default to 10 results
    
    try:
        # Generate embedding for the search term
        query_vector = generate_text_embedding(search_term)
        
        # Get the collection from Weaviate
        collection = weaviate_client.collections.get(collection_name)
        
        # Perform vector search
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit
        )
        
        # Format the results
        formatted_results = []
        for obj in results.objects:
            formatted_results.append({
                'id': obj.uuid,
                'name': obj.properties.get('name'),
                'path': obj.properties.get('path'),
                'folder': obj.properties.get('folder'),
                'distance': obj.metadata.distance
            })
        
        return jsonify({
            'query': search_term,
            'results': formatted_results
        })
    
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Weaviate Vector Search Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 5000)')
    parser.add_argument('--collection', default='space', help='Weaviate collection name')
    
    args = parser.parse_args()
    
    load_dotenv()

    # Best practice: store your credentials in environment variables
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
 
    
    # Set collection name
    collection_name = args.collection
    print(f"Using collection: {collection_name}")
    
    # Start the Flask server
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
