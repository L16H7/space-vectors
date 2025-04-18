import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
import os
from dotenv import load_dotenv

from clip import generate_image_embeddings

load_dotenv()

# Best practice: store your credentials in environment variables
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

print(client.is_ready())  # Should print: `True`

# space_collection = client.collections.create(
#     name="space",
# )

image_embeddings = generate_image_embeddings("/Users/light/hackathons/space-vectors/1.jpg")

space = client.collections.get("space")
uuid = space.data.insert(
    properties={
        "name": "space"
    },
    vector=image_embeddings
)

print(uuid)  # the return value is the object's UUID

# print(image_embeddings)

client.close()  # Free up resources
