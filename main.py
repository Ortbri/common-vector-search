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
    response = supabase.table("elements").select("thumbnail_url, element_id, clip_embedding").execute()
    
    if response.data is None:
        print("❌ No elements found in Supabase. Check table data.")
        return
    
    print(f"Found {len(response.data)} elements")
    
    for i, element in enumerate(response.data, 1):
        element_id = element.get("element_id")
        image_url = element.get("thumbnail_url")
        current_embedding = element.get("clip_embedding")

        if not element_id or not image_url:
            print(f"❌ Skipping element {i} - Missing required fields")
            continue
        
        print(f"\nProcessing element {i} of {len(response.data)}")
        print(f"Element ID: {element_id}")
        print(f"Image URL: {image_url}")
        print(f"Current embedding in DB: {current_embedding is not None}")
        
        # Download and process image
        image_path = "./temp.jpg"
        download_result = os.system(f"curl -s -f -o {image_path} '{image_url}'")
        
        if download_result != 0:
            print(f"❌ Failed to download image from {image_url}")
            continue

        if os.path.exists(image_path):
            print("✅ Image downloaded successfully")
            try:
                embedding = transform_image(image_path)
                
                if embedding is None or len(embedding) == 0:
                    print("❌ Generated embedding is empty")
                    continue

                print(f"Embedding shape: {embedding.shape}")
                print(f"Embedding sample values: {embedding[:5]}")  # Print first 5 values
                
                # Update Supabase
                print(f"Updating element {element_id} in Supabase...")
                update_data = {"clip_embedding": embedding.tolist()}
                print(f"Update payload size: {len(str(update_data))} bytes")
                print(f"Embedding dimension: {len(embedding)}")  # Should be 512
                
                try:
                    update_response = supabase.table("elements").update(
                        update_data
                    ).eq("element_id", element_id).execute()
                    print(f"Update response: {update_response}")
                except Exception as e:
                    print(f"❌ Update failed with error: {str(e)}")
                
                # Verify update
                try:
                    verify_response = supabase.table("elements").select(
                        "clip_embedding"
                    ).eq("element_id", element_id).execute()
                    
                    if verify_response.data and verify_response.data[0].get("clip_embedding"):
                        print(f"✅ Successfully verified embedding update for {element_id}")
                        print(f"Stored embedding: {verify_response.data[0]['clip_embedding'][:5]}")  # First 5 values
                    else:
                        print(f"❌ Failed to verify embedding update for {element_id}")
                        print(f"Verify response: {verify_response.data}")
                        print(f"Response data type: {type(verify_response.data[0].get('clip_embedding'))}")
                except Exception as e:
                    print(f"❌ Verification failed with error: {str(e)}")
                
            except Exception as e:
                print(f"❌ Error processing embedding: {str(e)}")
                continue
            finally:
                # Clean up temp file
                if os.path.exists(image_path):
                    os.remove(image_path)
        else:
            print(f"❌ Failed to download image from {image_url}")

if __name__ == "__main__":
    update_elements_with_embeddings()
