

## 👗 AI-Powered Product & Vibe Detection for Fashion Reels


## 📂 **Project Structure**

flickd_hackathon_submission/
├── app/
│   ├── api.py              # FastAPI app for serving predictions
│   ├── pipeline.py         # Batch and single inference pipeline
│   ├── utils.py            # Helper functions (embedding, cropping, etc.)
│   ├── visualization.py    # Data visualization utilities
├── models/
│   ├── yolov8n.pt          # YOLOv8 weights (for detection)
│   ├── catalog_embeds.npy  # Catalog product embeddings (CLIP)
├── catalog.xlsx            # Product catalog (Excel/CSV)
├── images.csv              # Catalog image mapping
├── vibeslist.json          # Supported fashion vibes
├── frames/                 # Sample video frames for demo/testing
├── labels/                 # YOLO detection labels (txt)
├── captions/               # Video captions (txt, optional)
├── outputs/                # Pipeline output JSONs (one per video)
├── requirements.txt        # pip dependencies
├── environment.yml         # Conda environment file
├── demo.mp4                # (Optional) Demo screen recording
├── README.md               # You are here!
└── .gitignore
```


## 🚀 **Quickstart**

### 1. **Clone & Set Up Environment**

```bash
git clone https://github.com/itachixtanjiro/flickd_hackathon_submission.git
cd flickd_hackathon_submission
conda env create -f environment.yml
conda activate flickd
```

### 2. **Download/Place All Assets**

* Download and place `yolov8n.pt`, `catalog_embeds.npy`, sample frames, and labels in their respective folders as above.
* Place provided `catalog.xlsx`, `images.csv`, and `vibeslist.json` at project root.

### 3. **Run Demo Pipeline**

```bash
python main.py
```

* Processes a sample frame and prints & saves predictions to `outputs/`.

### 4. **Run FastAPI Inference Server**

```bash
uvicorn app.api:app --host 0.0.0.0 --port 7860
```

* Visit [http://localhost:7860/docs](http://localhost:7860/docs) to test interactively.

---

## 🧠 **How It Works**

1. **Detection:** YOLOv8 detects clothing items in each frame (using label files).
2. **Cropping:** Each detected item is cropped out for matching.
3. **CLIP Matching:** Cropped images are embedded using CLIP and matched (via FAISS) to the closest catalog product.
4. **Vibe Tagging:** Video captions are classified to top-1/2/3 vibes using a transformer (if available) or fallback rules.
5. **Output:** For each video, an output JSON is created with the top vibes and matched products.

---

## 📝 **Sample Output (`outputs/sample_frame.json`)**

```json
{
  "video_id": "sample_frame",
  "vibes": ["Coquette", "Cottagecore"],
  "products": [
    {
      "matched_product_id": 12345,
      "type": "Top",
      "color": "White",
      "match_type": "exact",
      "confidence": 0.91,
      "detection_confidence": 0.84
    }
  ]
}
```

---

## 📊 **Visualization**

* Run:

  ```python
  from app.visualization import plot_product_types, plot_confidence_histogram
  plot_product_types(result)
  plot_confidence_histogram(result)
  ```
* See bar charts of predicted types and confidence scores.

---

## 📦 **Assets Provided**

* **yolov8n.pt**: YOLOv8 detection weights (pretrained or fine-tuned)
* **catalog\_embeds.npy**: Catalog product image embeddings (CLIP)
* **catalog.xlsx** & **images.csv**: Catalog and image mapping
* **vibeslist.json**: All supported fashion vibes/styles
* **frames/** & **labels/**: Sample frames and detections for demo
* **outputs/**: Pipeline JSONs

> **Note:** Training code and raw datasets are omitted for brevity, but will be provided upon request.

---

## 🏗️ **Reproducibility**

* All dependencies listed in `requirements.txt` and `environment.yml`
* All weights and catalog files included, or download instructions provided in this README

---

## 🎬 **Demo**

* See `demo.mp4` or visit \[https://drive.google.com/file/d/1p3PyV-4ck2MEevRNhWjBxJYQt_vlg-SZ/view?usp=sharing] for a live walkthrough of the pipeline and API usage.

---

## 👩‍💻 **API Example**

* **Start server:** `uvicorn app.api:app --host 0.0.0.0 --port 7860`
* **Test with Python:**

  ```python
  import requests
  files = {'image': open('frames/sample_frame.jpg', 'rb')}
  data = {'caption': "Summer dress with floral print"}
  r = requests.post('http://localhost:7860/predict/', files=files, data=data)
  print(r.json())
  ```

---

## 🛠️ **How to Train (If Custom YOLO/CLIP used)**

* Briefly: “We fine-tuned YOLOv8n on 200 fashion images using Ultralytics CLI for 10 epochs. Code available upon request.”

---

## 🙏 **Acknowledgements**

* YOLOv8 by Ultralytics
* OpenCLIP by LAION/ML Collective
* Hugging Face Transformers for vibe classification

---

## 📣 **Questions?**

Open an issue or email \[priyanshubadhan2228@gmail.com].

---

**Good luck, and thank you for reviewing our Flickd hackathon solution!**

