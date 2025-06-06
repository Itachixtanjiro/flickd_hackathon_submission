# main.py
import os
from app.utils import load_catalog, load_vibes, load_catalog_embeds, setup_clip_faiss
from app.pipeline import process_video
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ---- Paths (Update these as needed) ----
catalog_path = "product_data.xlsx"
images_path = "images.csv"
vibes_path = "vibeslist.json"
catalog_embeds_path = "models/catalog_embeds.npy"
frame_path = "frames/2025-05-31_14-01-37_UTC.jpg"
label_path = "labels/2025-05-31_14-01-37_UTC.txt"
caption_path = "captions/2025-05-31_14-01-37_UTC.txt"  # If you have captions

video_id = "2025-05-31_14-01-37_UTC"

# ---- Load all resources ----
catalog_df = load_catalog(catalog_path, images_path)
vibes_list = load_vibes(vibes_path)
catalog_embeds = load_catalog_embeds(catalog_embeds_path)
clip_model, preprocess, index = setup_clip_faiss(catalog_embeds)

with open(caption_path, 'r', encoding='utf-8', errors='ignore') as f:
    caption = f.read()


# ---- Run pipeline ----
result = process_video(
    video_id,
    frame_path,
    label_path,
    caption,
    catalog_df,
    index,
    preprocess,
    clip_model,
    vibes_list
)

# ---- Print and save result ----
import json
print(json.dumps(result, indent=2))

os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{video_id}.json", "w") as f:
    json.dump(result, f, indent=2)

# ---- Optional: Visualization ----
from app.visualization import plot_product_types, plot_confidence_histogram
plot_product_types(result)
plot_confidence_histogram(result)
