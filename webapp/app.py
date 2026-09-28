"""
webapp/app.py
=============
Medical Diagnose — Flask Inference Server
CNN + NLP + PEPA Medical Diagnosis Pipeline
PyTorch Backend — RTX 5050 / CUDA 12.8

Endpoints:
  GET  /        → home page        (index.html)
  POST /predict → run diagnosis    (result.html)
  GET  /predict → redirect to home
  GET  /health  → JSON status check

Model:
  Loads model_best.pth (PyTorch state dict)
  Architecture: EfficientNetB0 + NLP Dense + PEPA 
  Device: CUDA (RTX 5050) if available, else CPU

Image handling:
  Patient uploads X-ray / skin photo / scan / wound
  Supported: JPG, JPEG, PNG, BMP, WEBP — max 16 MB
  Preprocessed to (1, 3, 224, 224) tensor normalised [0,1]
  ImageNet normalisation: mean=[0.485,0.456,0.406]
                          std =[0.229,0.224,0.225]
  No image uploaded → zero tensor → PEPA ignores CNN

Authors:
  Palak Gautam         — pg3744@srmist.edu.in
Guide:
  Dr. Madhuri Sharma   — madhuris@srmist.edu.in
  SRM IST Delhi-NCR Campus
"""

import os
import re
import io
import csv
import pickle
import logging
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify
)
from werkzeug.utils import secure_filename

# ══════════════════════════════════════════════════════════════════
# ── LOGGING ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# ── PATHS ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
UPLOAD_DIR    = os.path.join(BASE_DIR, "webapp", "static", "uploads")
TEMPLATES_DIR = os.path.join(BASE_DIR, "webapp", "templates")
STATIC_DIR    = os.path.join(BASE_DIR, "webapp", "static")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_FILE_SIZE_MB   = 16
IMG_SIZE           = (224, 224)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# ── FLASK APP ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024
app.config["SECRET_KEY"] = os.urandom(24)


def load_homepage_accuracy(default_top1="87.0", default_top3="97.14"):
    """Load homepage Top-1/Top-3 values from A7 row only."""
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    if not os.path.exists(csv_path):
        return default_top1, default_top3

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return default_top1, default_top3

        # Use A7 row only.
        picked = next((r for r in rows if str(r.get("variant", "")).strip().upper() == "A7"), None)
        if picked is None:
            return default_top1, default_top3

        top1 = f"{float(picked.get('top1', default_top1)):.2f}"
        top3 = f"{float(picked.get('top3', default_top3)):.2f}"
        return top1, top3
    except Exception as e:
        log.warning(f"Failed to load homepage accuracy from ablation_results.csv: {e}")
        return default_top1, default_top3


HOME_TOP1, HOME_TOP3 = load_homepage_accuracy()


@app.context_processor
def inject_homepage_metrics():
    return {
        "top1_accuracy": HOME_TOP1,
        "top3_accuracy": HOME_TOP3,
    }

# ══════════════════════════════════════════════════════════════════
# ── DEVICE SETUP ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device : {device}")
if torch.cuda.is_available():
    log.info(f"GPU    : {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

# ══════════════════════════════════════════════════════════════════
# ── MODEL ARCHITECTURE (must match train.ipynb exactly) ───────────
# ══════════════════════════════════════════════════════════════════

class NLPEncoder(nn.Module):
    def __init__(self, input_dim=500, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, embed_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=128, freeze_base=True):
        super().__init__()
        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.features = base.features
        self.avgpool  = base.avgpool
        if freeze_base:
            for p in self.features.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, embed_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.head(self.avgpool(self.features(x)))


class PEPAModule(nn.Module):
    def __init__(self, img_dim=128, txt_dim=128, hidden_dim=64):
        super().__init__()
        self.gate_h   = nn.Linear(img_dim + txt_dim, hidden_dim)
        self.gate_o   = nn.Linear(hidden_dim, img_dim)
        self.txt_proj = nn.Linear(txt_dim, img_dim)
        self.relu     = nn.ReLU()
        self.sigmoid  = nn.Sigmoid()
    def forward(self, f_img, f_txt):
        z     = torch.cat([f_img, f_txt], dim=1)
        h     = self.relu(self.gate_h(z))
        alpha = self.sigmoid(self.gate_o(h))
        w_img = alpha * f_img
        w_txt = (1 - alpha) * self.relu(self.txt_proj(f_txt))
        return w_img + w_txt


class MedAIPipeline(nn.Module):
    def __init__(self, num_classes, tfidf_dim=500,
                 embed_dim=128, use_dropout=True, freeze_cnn=True):
        super().__init__()
        self.cnn_enc = CNNEncoder(embed_dim=embed_dim, freeze_base=freeze_cnn)
        self.nlp_enc = NLPEncoder(input_dim=tfidf_dim, embed_dim=embed_dim)
        self.pepa    = PEPAModule(img_dim=embed_dim, txt_dim=embed_dim)
        head = [
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU()
        ]
        if use_dropout:
            head.append(nn.Dropout(0.4))
        head += [
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        ]
        self.classifier = nn.Sequential(*head)

    def forward(self, img, text):
        f_img  = self.cnn_enc(img)
        f_txt  = self.nlp_enc(text)
        fused  = self.pepa(f_img, f_txt)
        return self.classifier(fused)

# ══════════════════════════════════════════════════════════════════
# ── LOAD ARTIFACTS ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

log.info("Loading artifacts...")

# 1. Load model config
try:
    with open(os.path.join(ARTIFACTS_DIR, "model_config.pkl"), "rb") as f:
        model_config = pickle.load(f)
    log.info(f"  model_config: {model_config}")
except Exception as e:
    log.warning(f"  model_config.pkl not found, using defaults: {e}")
    model_config = {
        "num_classes": 100,
        "tfidf_dim":   500,
        "embed_dim":   128,
        "use_dropout": True,
    }

# 2. Load model weights
model = None
try:
    model = MedAIPipeline(
        num_classes = model_config["num_classes"],
        tfidf_dim   = model_config["tfidf_dim"],
        embed_dim   = model_config["embed_dim"],
        use_dropout = model_config["use_dropout"],
    ).to(device)

    state = torch.load(
        os.path.join(ARTIFACTS_DIR, "model_best.pth"),
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()                   # inference mode — disables dropout/batchnorm training behaviour
    log.info(f"  model_best.pth loaded on {device}")
except Exception as e:
    log.error(f"  Failed to load model: {e}")
    model = None

# 3. Load encoder
encoder = None
try:
    with open(os.path.join(ARTIFACTS_DIR, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    log.info(f"  encoder.pkl  — {len(encoder.classes_)} disease classes")
except Exception as e:
    log.error(f"  Failed to load encoder: {e}")

# 4. Load vectorizer
vectorizer = None
try:
    with open(os.path.join(ARTIFACTS_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    log.info("  vectorizer.pkl loaded")
except Exception as e:
    log.error(f"  Failed to load vectorizer: {e}")

# 5. Load symptom columns
symptom_cols = []
try:
    with open(os.path.join(ARTIFACTS_DIR, "symptom_cols.pkl"), "rb") as f:
        symptom_cols = pickle.load(f)
    log.info(f"  symptom_cols.pkl — {len(symptom_cols)} symptom columns")
except Exception as e:
    log.warning(f"  symptom_cols.pkl not found: {e}")

# ══════════════════════════════════════════════════════════════════
# ── IMAGE PREPROCESSING ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

# Same transform as used in train.ipynb Dataset class
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])

# ══════════════════════════════════════════════════════════════════
# ── SYMPTOM ALIAS DICTIONARY ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

ALIASES = {
    "shortness of breath":           ["breathless", "cant breathe", "difficulty breathing",
                                      "breathing problem", "out of breath", "dyspnea"],
    "cough":                          ["dry cough", "wet cough", "persistent cough", "coughing"],
    "breathing fast":                 ["rapid breathing", "fast breathing", "hyperventilation"],
    "chest tightness":                ["tight chest", "pressure in chest", "chest heaviness"],
    "headache":                       ["head pain", "head ache", "migraine",
                                       "head hurts", "head is pounding"],
    "sharp chest pain":               ["chest pain", "chest ache", "heart pain",
                                       "angina", "stabbing chest"],
    "abdominal pain":                 ["stomach pain", "belly pain", "tummy ache",
                                       "stomach ache", "gut pain", "stomach cramps"],
    "back pain":                      ["lower back pain", "spine pain", "backache", "back hurts"],
    "joint pain":                     ["joint ache", "arthralgia", "bone pain", "joints hurt"],
    "sore throat":                    ["throat pain", "pharyngitis", "throat hurts"],
    "dizziness":                      ["dizzy", "lightheaded", "giddy", "vertigo", "spinning"],
    "seizures":                       ["seizure", "convulsions", "fits", "epileptic attack"],
    "disturbance of memory":          ["memory loss", "forgetful", "cant remember",
                                       "forget things", "poor memory"],
    "focal weakness":                 ["arm weakness", "leg weakness", "face drooping",
                                       "limb weakness", "one side weakness"],
    "slurring words":                 ["slurred speech", "cant speak properly", "garbled speech"],
    "difficulty speaking":            ["speech difficulty", "trouble speaking",
                                       "aphasia", "cant talk"],
    "abnormal involuntary movements": ["tremors", "trembling", "uncontrolled movements",
                                       "twitching", "shaking"],
    "insomnia":                       ["cant sleep", "sleep problems", "sleeplessness",
                                       "trouble sleeping"],
    "fever":                          ["high temperature", "high temp", "pyrexia",
                                       "hot body", "febrile", "feverish"],
    "fatigue":                        ["tired", "tiredness", "exhaustion", "lethargy",
                                       "no energy", "worn out", "always tired"],
    "weakness":                       ["weak", "no strength", "body weakness", "feeling weak"],
    "decreased appetite":             ["loss of appetite", "not hungry", "no appetite",
                                       "not eating", "anorexia"],
    "nausea":                         ["feel sick", "queasy", "nauseated", "sick to stomach"],
    "vomiting":                       ["throwing up", "puking", "being sick", "retching"],
    "diarrhea":                       ["loose stools", "watery stool", "loose motions",
                                       "diarrhoea"],
    "depression":                     ["depressed", "feeling low", "sadness", "hopeless",
                                       "no motivation", "feeling down"],
    "anxiety and nervousness":        ["anxious", "nervous", "nervousness", "panic",
                                       "worried", "on edge", "restless"],
    "rash":                           ["skin rash", "hives", "itchy skin", "eruption",
                                       "red spots", "skin spots", "breakout"],
    "itching of skin":                ["itching", "itchy", "skin itching", "pruritus"],
    "blurred vision":                 ["blurry vision", "vision problems", "cant see clearly",
                                       "fuzzy vision", "double vision"],
    "hoarse voice":                   ["hoarseness", "raspy voice", "voice change",
                                       "lost voice", "husky voice"],
    "nasal congestion":               ["blocked nose", "stuffy nose", "runny nose", "sneezing"],
    "palpitations":                   ["heart racing", "heart fluttering", "fast heartbeat",
                                       "heart pounding", "heart skipping"],
    "irregular heartbeat":            ["arrhythmia", "heart irregularity", "uneven heartbeat"],
}

# ══════════════════════════════════════════════════════════════════
# ── EMERGENCY KEYWORDS ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

EMERGENCY_KEYWORDS = [
    "chest pain", "sharp chest pain", "heart attack",
    "shortness of breath", "difficulty breathing", "cant breathe",
    "loss of consciousness", "unconscious", "fainted", "passed out",
    "severe bleeding", "heavy bleeding",
    "stroke", "face drooping", "arm weakness", "speech difficulty",
    "sudden severe headache", "worst headache",
    "paralysis", "unable to move", "numbness in face",
    "seizure", "convulsion", "fitting",
    "throat closing", "anaphylaxis",
]

# ══════════════════════════════════════════════════════════════════
# ── SPECIALIST MAP ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

SPECIALIST_MAP = {
    "heart":        "Cardiologist",      "cardiac":      "Cardiologist",
    "arrhythmia":   "Cardiologist",      "angina":       "Cardiologist",
    "asthma":       "Pulmonologist",     "lung":         "Pulmonologist",
    "pneumonia":    "Pulmonologist",     "bronchitis":   "Pulmonologist",
    "pulmonary":    "Pulmonologist",     "tuberculosis": "Pulmonologist",
    "covid":        "Pulmonologist",     "emphysema":    "Pulmonologist",
    "brain":        "Neurologist",       "neurol":       "Neurologist",
    "migraine":     "Neurologist",       "epilepsy":     "Neurologist",
    "seizure":      "Neurologist",       "stroke":       "Neurologist",
    "parkinson":    "Neurologist",       "alzheimer":    "Neurologist",
    "dementia":     "Neurologist",       "meningioma":   "Neurologist",
    "depression":   "Psychiatrist",      "anxiety":      "Psychiatrist",
    "bipolar":      "Psychiatrist",      "schizophrenia":"Psychiatrist",
    "panic":        "Psychiatrist",      "psychosis":    "Psychiatrist",
    "ptsd":         "Psychiatrist",      "insomnia":     "Psychiatrist",
    "gastro":       "Gastroenterologist","liver":        "Gastroenterologist",
    "stomach":      "Gastroenterologist","colon":        "Gastroenterologist",
    "hepatitis":    "Gastroenterologist","cirrhosis":    "Gastroenterologist",
    "diabetes":     "Endocrinologist",   "thyroid":      "Endocrinologist",
    "hormone":      "Endocrinologist",   "adrenal":      "Endocrinologist",
    "hypoglycemia": "Endocrinologist",
    "skin":         "Dermatologist",     "dermat":       "Dermatologist",
    "eczema":       "Dermatologist",     "psoriasis":    "Dermatologist",
    "melanoma":     "Dermatologist",     "acne":         "Dermatologist",
    "kidney":       "Nephrologist",      "renal":        "Nephrologist",
    "cystitis":     "Urologist",         "bladder":      "Urologist",
    "urinary":      "Urologist",         "prostate":     "Urologist",
    "eye":          "Ophthalmologist",   "vision":       "Ophthalmologist",
    "cataract":     "Ophthalmologist",   "glaucoma":     "Ophthalmologist",
    "retinopathy":  "Ophthalmologist",
    "joint":        "Rheumatologist",    "arthritis":    "Rheumatologist",
    "fibromyalgia": "Rheumatologist",    "lupus":        "Rheumatologist",
    "fracture":     "Orthopaedic Surgeon","bone":        "Orthopaedic Surgeon",
    "spondyl":      "Orthopaedic Surgeon",
    "cancer":       "Oncologist",        "tumor":        "Oncologist",
    "lymphoma":     "Oncologist",        "leukemia":     "Oncologist",
    "vaginal":      "Gynaecologist",     "uterine":      "Gynaecologist",
    "ovarian":      "Gynaecologist",     "menstrual":    "Gynaecologist",
    "pregnancy":    "Obstetrician",      "endometri":    "Gynaecologist",
    "malaria":      "Infectious Disease Specialist",
    "typhoid":      "Infectious Disease Specialist",
    "sepsis":       "Infectious Disease Specialist",
    "throat":       "ENT Specialist",    "sinus":        "ENT Specialist",
    "nasal":        "ENT Specialist",    "tonsil":       "ENT Specialist",
    "hemorrhage":   "Emergency Medicine","poisoning":    "Emergency Medicine",
}

# ══════════════════════════════════════════════════════════════════
# ── HELPER FUNCTIONS ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def is_model_ready() -> bool:
    return all([
        model     is not None,
        encoder   is not None,
        vectorizer is not None,
    ])


def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def recommend_specialist(disease: str) -> str:
    dl = disease.lower()
    for keyword, spec in SPECIALIST_MAP.items():
        if keyword in dl:
            return spec
    return "General Physician"


def check_emergency(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in EMERGENCY_KEYWORDS)


def parse_symptoms(user_input: str):
    """
    Parse raw symptom text.
    Returns:
        clean_text : preprocessed string for TF-IDF
        detected   : list of matched symptom names for display
    """
    clean    = re.sub(r"[^a-zA-Z\s,]", " ", user_input.lower().strip())
    clean    = re.sub(r"\s+", " ", clean).strip()
    detected = []

    for col in symptom_cols:
        col_clean = col.replace("_", " ").lower().strip()
        if col_clean in clean:
            detected.append(col.replace("_", " ").title())
            continue
        for alias in ALIASES.get(col_clean, []):
            if alias in clean:
                detected.append(col.replace("_", " ").title())
                break

    return clean, list(set(detected))


def preprocess_image(file_storage) -> torch.Tensor:
    """
    Preprocess uploaded image for CNN encoder.

    Steps:
        1. Read bytes from upload
        2. Open with PIL and convert to RGB
        3. Apply torchvision transform:
           Resize(224,224) → ToTensor → Normalize(ImageNet)
        4. Add batch dimension → (1, 3, 224, 224)

    Returns:
        torch.Tensor (1, 3, 224, 224) on CPU
        Falls back to zero tensor on any error
    """
    try:
        img_bytes = file_storage.read()
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor    = IMG_TRANSFORM(img).unsqueeze(0)   # (1, 3, 224, 224)
        return tensor
    except Exception as e:
        log.warning(f"Image preprocessing failed: {e} — using zero tensor")
        return torch.zeros(1, 3, *IMG_SIZE)


def get_zero_image() -> torch.Tensor:
    """
    Zero tensor for text-only inference.
    PEPA gate learns α → 0 when this is the input,
    routing prediction entirely through the NLP branch.
    """
    return torch.zeros(1, 3, *IMG_SIZE)


@torch.no_grad()
def run_inference(clean_text: str, img_tensor: torch.Tensor, top_k: int = 3):
    """
    Run CNN + NLP + PEPA inference.

    Args:
        clean_text  : preprocessed symptom string
        img_tensor  : (1, 3, 224, 224) — real or zeros
        top_k       : number of top predictions to return

    Returns:
        list of (disease_name, confidence_pct) tuples
    """
    # TF-IDF vectorise
    tfidf_vec = vectorizer.transform([clean_text]).toarray().astype(np.float32)
    text_t    = torch.tensor(tfidf_vec).to(device)
    img_t     = img_tensor.to(device)

    # Forward pass
    logits = model(img_t, text_t)                      # (1, num_classes)
    probs  = torch.softmax(logits, dim=1)[0]           # (num_classes,)

    # Top-K
    top_probs, top_idxs = torch.topk(probs, k=top_k)

    results = [
        (
            encoder.inverse_transform([idx.item()])[0].title(),
            round(prob.item() * 100, 2)
        )
        for prob, idx in zip(top_probs, top_idxs)
    ]
    return results

# ══════════════════════════════════════════════════════════════════
# ── ROUTES ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """Home page with symptom input form."""
    if not is_model_ready():
        return render_template(
            "index.html",
            error="Model not loaded. Run train.ipynb first to generate model_best.pth.",
            symptom_list=[]
        ), 503

    chip_list = [col.replace("_", " ").title() for col in symptom_cols[:80]]
    return render_template("index.html", symptom_list=chip_list)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Main prediction endpoint."""

    chip_list = [col.replace("_", " ").title() for col in symptom_cols[:80]]

    if request.method == "GET":
        return redirect(url_for("home"))

    # Validate model
    if not is_model_ready():
        return render_template(
            "index.html",
            error="Model not ready. Please contact the administrator.",
            symptom_list=chip_list
        ), 503

    # Read form inputs
    user_input = request.form.get("symptoms", "").strip()
    image_file = request.files.get("medical_image")

    if not user_input or len(user_input) < 3:
        return render_template(
            "index.html",
            error="Please describe your symptoms in more detail.",
            symptom_list=chip_list
        )

    # Parse symptoms
    clean_text, detected = parse_symptoms(user_input)
    log.info(f"Symptoms parsed: '{user_input[:60]}' → {len(detected)} matched")

    # Process image
    image_used = False
    image_name = None
    img_tensor = get_zero_image()

    if image_file and image_file.filename and image_file.filename != "":
        if allowed_file(image_file.filename):
            img_tensor = preprocess_image(image_file)
            image_used = True
            image_name = secure_filename(image_file.filename)
            log.info(f"Image processed: {image_name}")
        else:
            log.warning(f"Unsupported file type: {image_file.filename}")

    # Run inference
    try:
        top3 = run_inference(clean_text, img_tensor, top_k=3)
        log.info(f"Prediction: {top3[0][0]} ({top3[0][1]}%) | image_used={image_used}")
    except Exception as e:
        log.error(f"Inference failed: {e}")
        return render_template(
            "index.html",
            error="Diagnosis failed. Please try again.",
            symptom_list=chip_list
        ), 500

    # Post-processing
    is_emergency     = check_emergency(user_input)
    specialist       = recommend_specialist(top3[0][0])
    top1_conf        = top3[0][1]

    if top1_conf >= 60:
        confidence_level = "High"
    elif top1_conf >= 30:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    return render_template(
        "result.html",
        top3              = top3,
        specialist        = specialist,
        emergency         = is_emergency,
        detected_symptoms = detected,
        image_used        = image_used,
        image_name        = image_name,
        user_input        = user_input,
        confidence_level  = confidence_level,
        num_diseases      = len(encoder.classes_) if encoder else 0
    )


@app.route("/health")
def health():
    """Health check — returns JSON system status."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name":   torch.cuda.get_device_name(0),
            "vram_total": f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB",
            "vram_used":  f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
        }

    status = {
        "status":            "ok" if is_model_ready() else "degraded",
        "device":            str(device),
        "model_loaded":      model is not None,
        "encoder_loaded":    encoder is not None,
        "vectorizer_loaded": vectorizer is not None,
        "symptom_cols":      len(symptom_cols),
        "disease_classes":   len(encoder.classes_) if encoder else 0,
        **gpu_info
    }
    return jsonify(status), 200 if is_model_ready() else 503


# ── ERROR HANDLERS ────────────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(e):
    chip_list = [col.replace("_", " ").title() for col in symptom_cols[:80]]
    return render_template(
        "index.html",
        error=f"Image too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        symptom_list=chip_list
    ), 413


@app.errorhandler(404)
def not_found(e):
    return redirect(url_for("home"))


@app.errorhandler(500)
def server_error(e):
    chip_list = [col.replace("_", " ").title() for col in symptom_cols[:80]]
    return render_template(
        "index.html",
        error="An internal error occurred. Please try again.",
        symptom_list=chip_list
    ), 500


# ══════════════════════════════════════════════════════════════════
# ── ENTRY POINT ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 55)
    log.info("  Medical Diagnose — Flask Server Starting")
    log.info("=" * 55)
    log.info(f"  Framework      : PyTorch {torch.__version__}")
    log.info(f"  Device         : {device}")
    log.info(f"  Model ready    : {is_model_ready()}")
    log.info(f"  Disease classes: {len(encoder.classes_) if encoder else 0}")
    log.info(f"  Symptom cols   : {len(symptom_cols)}")
    log.info(f"  URL            : http://127.0.0.1:5000")
    log.info("=" * 55)

    app.run(
        debug    = False,
        host     = "0.0.0.0",
        port     = 5000,
        threaded = True
    )
