import numpy as np
import pandas as pd
import faiss
import open_clip
import torch
from PIL import Image
import cv2

# ---- Data Loaders ----
def load_catalog(catalog_path, images_path):
    product_df = pd.read_excel(catalog_path)
    product_df.columns = product_df.columns.str.strip()
    images_df = pd.read_csv(images_path)
    images_df.columns = images_df.columns.str.strip()
    catalog_df = pd.merge(product_df, images_df, on='id', how='left')
    return catalog_df

def load_vibes(vibes_path):
    import json
    with open(vibes_path, 'r') as f:
        vibes_list = json.load(f)
    return vibes_list

def load_catalog_embeds(embeds_path):
    return np.load(embeds_path)

# ---- CLIP + FAISS Setup ----
def setup_clip_faiss(catalog_embeds, device="cpu"):
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device)
    EMBED_DIM = 512
    catalog_embeds = np.ascontiguousarray(catalog_embeds, dtype=np.float32)
    faiss.normalize_L2(catalog_embeds)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(catalog_embeds)
    return clip_model, preprocess, index

def embed_image(img, preprocess, clip_model, device="cpu"):
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img_t).cpu().numpy().flatten()
    return emb

# ---- YOLO Crop Helper ----
def crop_detections(frame_path, label_path, min_det_conf=0.25):
    frame = cv2.imread(frame_path)
    h, w, _ = frame.shape
    det_df = pd.read_csv(label_path, sep=' ', header=None)
    det_df.columns = ['class_id', 'x_center', 'y_center', 'width', 'height', 'conf']
    det_df = det_df[det_df['conf'] >= min_det_conf]
    crops = []
    for idx, row in det_df.iterrows():
        cx, cy, bw, bh = row['x_center'], row['y_center'], row['width'], row['height']
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        crop = frame[max(y1,0):min(y2,h), max(x1,0):min(x2,w)]
        if crop.size == 0: continue
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crops.append({'index': idx, 'img': pil_crop, 'conf': float(row['conf'])})
    return crops

# ---- Vibe Classification ----
def classify_vibe(caption, vibes_list, model=None):
    try:
        if model:
            result = model(caption, vibes_list)
            return [label for label, score in zip(result['labels'], result['scores']) if score > 0.2][:3]
    except Exception:
        pass
    cap = caption.lower()
    matches = [v for v in vibes_list if v.lower() in cap]
    if not matches:
        if 'summer' in cap or 'linen' in cap: matches.append('Cottagecore')
        if 'breezy' in cap or 'vest' in cap: matches.append('Coquette')
    return matches[:3] if matches else ['Coquette']

