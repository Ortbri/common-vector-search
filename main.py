import os
import torch # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
from torchvision import transforms # type: ignore
from transformers import CLIPModel # type: ignore
from supabase import create_client, Client # type: ignore
from dotenv import load_dotenv # type: ignore

# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials. Please check your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load CLIP model
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform_image(image_path):
    """Transform an image into a CLIP embedding."""
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model.get_image_features(pixel_values=image)
    embedding = features.squeeze().cpu().numpy()
    return embedding.astype(np.float32)

def update_elements_with_embeddings():
    """Retrieve elements from Supabase and update with vector embeddings."""
    print("\nFetching elements from Supabase...")
    response = supabase.table("elements").select("element_id, jpg_url").execute()
    elements = response.data

    for element in elements:
        element_id = element["element_id"]
        image_url = element["jpg_url"]

        if image_url:
            try:
                # Download and process image
                image_path = "./temp.jpg"
                os.system(f"curl -o {image_path} {image_url}")  # Download image
                embedding = transform_image(image_path)

                # Update Supabase with vector embedding
                supabase.table("elements").update({
                    "clip_embedding": embedding.tolist()
                }).eq("element_id", element_id).execute()

                print(f"✅ Updated {element_id}")

            except Exception as e:
                print(f"❌ Error processing {element_id}: {str(e)}")

if __name__ == "__main__":
    update_elements_with_embeddings()
