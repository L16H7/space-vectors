import os
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import glob
from pathlib import Path
from tqdm import tqdm
import argparse
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
import os
from dotenv import load_dotenv



# This should be your actual client import
# For example:
# from your_vector_db_library import Client
# Replace this with your actual vector DB client import

def setup_model():
    """Setup the SigLIP model and processor."""
    print("Loading SigLIP model and processor...")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-384")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
    return model, processor

def generate_image_embeddings(image_path, model, processor):
    """Generate image embeddings for a given image path."""
    try:
        # Load and process the image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        
        # Return as a numpy array for storage
        return outputs[0].cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def find_image_files(directory):
    """Find all image files in the given directory and subdirectories."""
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    image_files = []
    
    # Walk through directory structure
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if file has image extension
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def process_directory(directory, collection_name, db_client):
    """Process all images in a directory and add them to the vector database."""
    # Setup model
    model, processor = setup_model()
    
    # Get collection
    space = db_client.collections.get(collection_name)
    
    # Find all image files
    print(f"Finding images in {directory}...")
    image_files = find_image_files(directory)
    print(f"Found {len(image_files)} images.")
    
    # Process each image
    results = []
    for image_path in tqdm(image_files, desc="Processing images"):
        # Generate a relative path for storing in the DB
        relative_path = os.path.relpath(image_path, directory)
        
        # Generate embeddings
        embeddings = generate_image_embeddings(image_path, model, processor)
        
        if embeddings is not None:
            # Insert into vector DB
            try:
                uuid = space.data.insert(
                    properties={
                        "name": relative_path,
                        "path": str(image_path),
                        "folder": os.path.dirname(relative_path)
                    },
                    vector=embeddings
                )
                results.append((image_path, uuid))
                print(f"Inserted {relative_path} with UUID: {uuid}")
            except Exception as e:
                print(f"Error inserting {image_path} into vector DB: {e}")
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Insert images into vector database.')
    parser.add_argument('directory', type=str, help='Root directory containing images')
    parser.add_argument('--collection', type=str, default='space', 
                        help='Name of vector DB collection')
    args = parser.parse_args()
    
    # This should be your actual client initialization
    # For example:
    # client = Client(...)
    # Replace with your actual client initialization code
    load_dotenv()

    # Best practice: store your credentials in environment variables
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    
    try:
        # Process the directory
        results = process_directory(args.directory, args.collection, client)
        
        # Print summary
        print(f"Successfully processed {len(results)} images.")
        
    finally:
        # Close client connection
        client.close()
        print("Vector DB client connection closed.")

if __name__ == "__main__":
    main()