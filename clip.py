import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

# Load the model and processor
model = AutoModel.from_pretrained("google/siglip-base-patch16-384")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")

# Load your image
def generate_image_embeddings(image_path):
    image = load_image(image_path)
    inputs = processor(images=[image], return_tensors="pt")

    # Generate image embeddings
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
        
    return image_embeddings
