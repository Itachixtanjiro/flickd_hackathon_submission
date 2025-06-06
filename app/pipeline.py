import os
from collections import defaultdict
import pandas as pd
from app.utils import (load_catalog, load_vibes, load_catalog_embeds,
                       setup_clip_faiss, embed_image, crop_detections, classify_vibe)

def process_video(video_id, frame_path, label_path, caption, catalog_df, index, preprocess, clip_model, vibes_list, vibe_model=None,
                  min_det_conf=0.25, min_clip_sim=0.75, top_k=3):
    # 1. Crop detected objects
    crops = crop_detections(frame_path, label_path, min_det_conf=min_det_conf)
    # 2. CLIP+FAISS match for each crop
    all_matches = []
    for crop in crops:
        emb = embed_image(crop['img'], preprocess, clip_model)
        import faiss
        D, I = index.search(emb.reshape(1, -1).astype('float32'), top_k)
        for sim, idx_prod in zip(D[0], I[0]):
            if sim < min_clip_sim: continue
            prod_row = catalog_df.iloc[idx_prod]
            match_type = 'exact' if sim > 0.9 else 'similar'
            all_matches.append({
                'det_index': crop['index'],
                'matched_product_id': int(prod_row['id']),
                'type': prod_row.get('product_type', ''),
                'color': prod_row.get('product_tags', ''),
                'match_type': match_type,
                'confidence': float(sim),
                'detection_confidence': crop['conf']
            })
    # 3. Deduplicate: one best per product per detection
    best_matches = defaultdict(lambda: {})
    for m in all_matches:
        det_idx, prod_id = m['det_index'], m['matched_product_id']
        if prod_id not in best_matches[det_idx] or m['confidence'] > best_matches[det_idx][prod_id]['confidence']:
            best_matches[det_idx][prod_id] = m
    final_matches = []
    for det_idx, prod_dict in best_matches.items():
        final_matches.extend(prod_dict.values())
    final_matches = sorted(final_matches, key=lambda x: (x['det_index'], -x['confidence']))
    # 4. Vibes
    vibes = classify_vibe(caption, vibes_list, model=vibe_model)
    # 5. Return output dict
    return {
        "video_id": video_id,
        "vibes": vibes,
        "products": [
            {
                'matched_product_id': m['matched_product_id'],
                'type': m['type'],
                'color': m['color'],
                'match_type': m['match_type'],
                'confidence': round(m['confidence'], 3),
                'detection_confidence': round(m['detection_confidence'], 3)
            }
            for m in final_matches
        ]
    }
