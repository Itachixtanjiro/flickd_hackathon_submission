from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from app.utils import (load_catalog, load_vibes, load_catalog_embeds,
                       setup_clip_faiss, embed_image, classify_vibe)

# ---- Load all models once on API startup ----
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
    img = Image.open(BytesIO(await image.read())).convert("RGB")
    emb = embed_image(img, preprocess, clip_model, device=device).astype('float32')
    import faiss
    faiss.normalize_L2(emb.reshape(1,-1))
    D, I = index.search(emb.reshape(1,-1), k=3)
    matches = []
    for sim, idx in zip(D[0], I[0]):
        if sim < 0.75:
            continue
        row = catalog_df.iloc[idx]
        matches.append({
            'matched_product_id': int(row['id']),
            'type': row.get('product_type', ''),
            'color': row.get('product_tags', ''),
            'confidence': float(sim)
        })
    vibes = classify_vibe(caption, vibes_list, model=vibe_model)
    return JSONResponse({
        "vibes": vibes,
        "products": matches
    })
