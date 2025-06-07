import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Suppress OpenMP warnings (Windows)
import faiss
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from app.utils import (
    load_catalog, load_vibes, load_catalog_embeds,
    setup_clip_faiss, embed_image, classify_vibe
)

# ---- Load resources ONCE on API startup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
catalog_df = load_catalog("product_data.xlsx", "images.csv")
vibes_list = load_vibes("vibeslist.json")
catalog_embeds = load_catalog_embeds("models/catalog_embeds.npy")
clip_model, preprocess, index = setup_clip_faiss(catalog_embeds, device=device)

try:
    from transformers import pipeline as hf_pipeline
    vibe_model = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception:
    vibe_model = None

app = FastAPI()

@app.post("/predict/")
async def predict(
    image: UploadFile = File(...),
    caption: str = Form(...)
):
    # Read image and preprocess for CLIP
    img = Image.open(BytesIO(await image.read())).convert("RGB")
    emb = embed_image(img, preprocess, clip_model, device=device).astype('float32')
    faiss.normalize_L2(emb.reshape(1, -1))
    D, I = index.search(emb.reshape(1, -1), k=3)

    # ---- DEBUG: print all sim scores and IDs ----
    print("Similarity scores (D):", D)
    print("Indexes (I):", I)
    for sim, idx in zip(D[0], I[0]):
        print("Product ID:", catalog_df.iloc[idx]['id'], "Sim score:", sim)

    # ---- Return top-3 products always (for debugging) ----
    matches = []
    for sim, idx in zip(D[0], I[0]):
        row = catalog_df.iloc[idx]
        matches.append({
            'matched_product_id': int(row['id']),
            'type': row.get('product_type', ''),
            'color': row.get('product_tags', ''),
            'confidence': float(sim)
        })

    # ---- For production, you can add a threshold filter here ----
    # e.g., only return matches with sim > 0.7
    # matches = [m for m in matches if m['confidence'] > 0.7]

    vibes = classify_vibe(caption, vibes_list, model=vibe_model)
    return JSONResponse({
        "vibes": vibes,
        "products": matches
    })

