# 🧠 XAI Emotion Recognition

**Beyond Heatmaps: Grounded Textual Explanations for Facial Emotion Recognition via Multimodal Feature Fusion and Vision-Language Models**

> A research-grade multimodal pipeline that produces **natural language explanations** for facial emotion predictions, grounded in Grad-ECLIP attention evidence and FACS-based facial feature analysis — making XAI truly autonomous and interpretable.

---

## 🔥 Key Innovation

Traditional XAI methods (SHAP, LIME, Grad-CAM) produce **visual explanations** (heatmaps) that require human experts to interpret. Our system **eliminates the human-in-the-loop** by generating clinical textual explanations that explain *why* a specific emotion was predicted.

```
Input: Face Image
Output:
  Emotion: Happy (87% confidence)
  
  Explanation: "The person appears happy because their lip corners are 
  pulled upward (AU12), cheeks are raised (AU6), and the model's attention 
  is strongly focused on the mouth and eye regions. These features are 
  consistent with the FACS indicators of genuine happiness (Duchenne smile)."
  
  Visual: Grad-ECLIP heatmap + Landmark overlay
```

---

## 📐 Architecture

```
Input Image
   ↓
MediaPipe Face Mesh (468 landmarks)
   ↓
┌──────────────────────────────────────────────────┐
│  Branch 1: AU Features     Branch 2: POSTER V2   │
│  (geometric distances)     (IR-50 + Cross-Attn)  │
│                                ↓                  │
│                            Grad-ECLIP             │
│                            (Attention Map)        │
│                                ↓                  │
│                            Emotion Logits         │
└──────────────────────────────────────────────────┘
   ↓                              ↓
   Feature Fusion (AUs + Attention Regions + Emotion + Confidence)
   ↓
   LLaVA-7B (4-bit quantized — Explanation Generator)
   ↓
   Final Output: {emotion, confidence, explanation, heatmap, landmarks}
```

---

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (tested on RTX 4050)
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation
```bash
# Clone the repository
git clone https://github.com/ad23b1012/MemoryPalAI.git
cd MemoryPalAI

# Install with uv
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Download FER2013 Dataset
```bash
# Via Kaggle CLI
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/

# Or download manually from:
# https://www.kaggle.com/datasets/msambare/fer2013
```

---

## 🚀 Usage

### 1. Train the Classifier
```bash
# Train POSTER V2 (primary model)
python scripts/train_classifier.py \
    --model poster_v2 \
    --dataset fer2013 \
    --data-path data/fer2013/fer2013.csv \
    --epochs 50 \
    --batch-size 32

# Train ResNet-50+CBAM (baseline for ablation)
python scripts/train_classifier.py \
    --model resnet50_cbam \
    --dataset fer2013 \
    --data-path data/fer2013/fer2013.csv \
    --epochs 50
```

### 2. Evaluate
```bash
python scripts/evaluate.py \
    --model poster_v2 \
    --dataset fer2013 \
    --data-path data/fer2013/fer2013.csv \
    --checkpoint checkpoints/poster_v2_best.pth \
    --split PrivateTest
```

### 3. Run Full XAI Pipeline
```bash
# Full pipeline with LLaVA-7B explanation
python scripts/demo.py --image path/to/face.jpg

# Without VLM (faster, classification + attention only)
python scripts/demo.py --image path/to/face.jpg --no-explanation
```

---

## 📊 Results

### Emotion Classification Accuracy

| Model | FER2013 (Public) | FER2013 (Private) | RAF-DB |
|-------|:---:|:---:|:---:|
| Aly et al. (ResNet+CBAM) | 88.13% | 73.43% | 87.62% |
| ResNet-50+CBAM (ours, reproduced) | TBD | TBD | TBD |
| **POSTER V2 (ours)** | **TBD** | **TBD** | **TBD** |

### VRAM Usage (RTX 4050 6GB)

| Phase | Component | Peak VRAM |
|-------|-----------|-----------|
| Phase 1 | POSTER V2 + Grad-ECLIP | ~2.5 GB |
| Phase 2 | LLaVA-7B (4-bit) | ~4.5 GB |

---

## 📁 Project Structure

```
├── pyproject.toml              # Dependencies & project config
├── configs/
│   └── default.yaml            # All hyperparameters
├── src/
│   ├── face_detection/
│   │   ├── detector.py         # MediaPipe Face Mesh wrapper
│   │   └── au_extractor.py     # Landmark → FACS Action Units
│   ├── emotion/
│   │   ├── model.py            # POSTER V2 classifier
│   │   ├── baseline_resnet_cbam.py  # ResNet-50+CBAM baseline
│   │   ├── dataset.py          # FER2013 / RAF-DB dataloaders
│   │   └── train.py            # Training loop (AMP, label smoothing)
│   ├── attention/
│   │   ├── grad_eclip.py       # Grad-ECLIP attention maps
│   │   └── region_parser.py    # Heatmap → semantic region labels
│   ├── explainer/
│   │   ├── prompt_builder.py   # Evidence-based prompt construction
│   │   └── vlm_engine.py       # LLaVA-7B 4-bit inference
│   ├── pipeline.py             # End-to-end orchestrator
│   └── visualization.py        # Publication-quality plots
├── scripts/
│   ├── train_classifier.py     # Training CLI
│   ├── evaluate.py             # Evaluation CLI
│   └── demo.py                 # Demo CLI
├── data/                       # Datasets (.gitignored)
├── checkpoints/                # Model weights (.gitignored)
└── outputs/                    # Results & visualizations
```

---

## 📝 Research Paper

**Title:** "Beyond Heatmaps: Grounded Textual Explanations for Facial Emotion Recognition via Multimodal Feature Fusion and Vision-Language Models"

### Key Contributions
1. **Novel pipeline:** First end-to-end system combining geometric AU features + Grad-ECLIP attention + VLM explanation
2. **Autonomous XAI:** No human expert needed to interpret model predictions
3. **Grad-ECLIP for FER:** First application of Grad-ECLIP to facial expression recognition
4. **Lightweight:** Runs on 6GB consumer GPU via sequential model loading

### Reference Papers
- Aly et al. (2023) — ResNet-50+CBAM for FER (IEEE Access)
- Zheng et al. (2023) — POSTER V2 (ICCV 2023)
- Zhao et al. (2024) — Grad-ECLIP (ICML 2024)

---

## ⚙️ Hardware Requirements
- **GPU:** NVIDIA RTX 4050 (6GB GDDR6) or equivalent
- **RAM:** 16GB+ recommended
- **Storage:** ~5GB for datasets + ~10GB for model weights

## License
MIT
