import streamlit as st
import numpy as np
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
from PIL import Image
import io

# ─── Label sets ──────────────────────────────────────────────────────────────
LABELS_29 = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'Delete', 'Nothing', 'Space'
]
LABELS_26 = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
             'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignSpeak – ASL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Quicksand:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Quicksand', sans-serif; }
header, footer { visibility: hidden; }

.stApp {
    background: linear-gradient(135deg, #fbf7ff 0%, #eef4ff 50%, #fff5fb 100%);
    min-height: 100vh;
}

.brand {
    font-family: 'Fredoka', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #2b3a55;
    letter-spacing: -0.5px;
}
.sub { color: #7b88a8; font-size: 1rem; font-weight: 600; }

.card {
    background: #ffffff;
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 10px 36px rgba(150,130,210,0.10);
    border: 1.5px solid #f0ecf8;
}

.result-box {
    background: linear-gradient(135deg, #eefbf4, #f0f7ff);
    border-radius: 20px;
    padding: 40px 20px;
    text-align: center;
    border: 2px dashed #b6e6cc;
    margin: 16px 0;
}
.result-letter {
    font-family: 'Fredoka', sans-serif;
    font-size: 5rem;
    font-weight: 700;
    color: #1b4d3e;
    line-height: 1;
}
.result-word {
    font-family: 'Fredoka', sans-serif;
    font-size: 1.5rem;
    color: #3a7d5e;
    margin-top: 8px;
    font-weight: 600;
}

.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 700;
    color: #5b6882;
    margin-top: 12px;
}

.info-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0f4ff;
    color: #4a5aaa;
    border-radius: 12px;
    padding: 6px 14px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 14px;
}

.tip-box {
    background: #fffbea;
    border: 1.5px solid #fde68a;
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 0.9rem;
    color: #92400e;
    margin-top: 14px;
}
.tip-box b { color: #78350f; }

.skeleton-img { border-radius: 14px; overflow: hidden; }

.stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #a855f7) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ASL model…")
def load_model():
    try:
        return keras.saving.load_model("best_cnn_asl_model.keras")
    except Exception:
        try:
            import tensorflow as tf
            return tf.keras.models.load_model("best_cnn_asl_model.keras")
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return None

# ─── Load MediaPipe Hand Landmarker ──────────────────────────────────────────
@st.cache_resource(show_spinner="Loading hand detector…")
def load_detector():
    """Returns (detector_type, detector). Tries Tasks API then legacy solutions."""
    task_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        if os.path.exists(task_file):
            base_opts = python.BaseOptions(model_asset_path=task_file)
            opts = vision.HandLandmarkerOptions(
                base_options=base_opts,
                num_hands=1,
                min_hand_detection_confidence=0.1,
                min_hand_presence_confidence=0.1,
                min_tracking_confidence=0.1
            )
            return "tasks", vision.HandLandmarker.create_from_options(opts)
    except Exception:
        pass
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            return "solutions", mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=1,
                min_detection_confidence=0.2
            )
    except Exception:
        pass
    return None, None


def detect_landmarks(dtype, detector, img_np):
    """Returns (coords [63 floats], raw_landmark_list) or ([], None)."""
    if dtype == "tasks":
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        res = detector.detect(mp_img)
        if res.hand_landmarks:
            lms = res.hand_landmarks[0]
            return [v for lm in lms for v in (lm.x, lm.y, lm.z)], lms
    elif dtype == "solutions":
        cpy = img_np.copy(); cpy.flags.writeable = False
        res = detector.process(cpy)
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            return [v for lm in lms for v in (lm.x, lm.y, lm.z)], lms
    return [], None


def annotate_hand(img_np, landmarks):
    """Draw hand skeleton on image for visual feedback."""
    import cv2
    out = img_np.copy()
    h, w = out.shape[:2]
    CONNS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
             (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
             (13,17),(17,18),(18,19),(19,20),(0,17)]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(out, pts[a], pts[b], (50, 220, 80), 2)
    for x, y in pts:
        cv2.circle(out, (x, y), 5, (255, 60, 60), -1)
        cv2.circle(out, (x, y), 6, (255, 255, 255), 1)
    return out


def predict(model, coords):
    """Run model inference and return (pred_char, confidence, top3 list)."""
    inp = np.array(coords, dtype=np.float32).reshape(1, 63, 1)
    preds = model.predict(inp, verbose=0)[0]
    
    nc = model.output_shape[-1]
    labels = LABELS_29 if nc == 29 else LABELS_26 if nc == 26 else [chr(ord('A') + i) for i in range(nc)]
    
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [(labels[i] if i < len(labels) else str(i), float(preds[i] * 100)) for i in top3_idx]
    pred_char = top3[0][0]
    confidence = top3[0][1]
    return pred_char, confidence, top3


# ─── Load resources ──────────────────────────────────────────────────────────
model = load_model()
detector_type, detector = load_detector()

# ─── Header ──────────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown('<div class="brand">🤟 SignSpeak</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">ASL Sign Language → English</div>', unsafe_allow_html=True)
with c2:
    st.markdown("")
    if detector_type:
        st.markdown(f'<div class="info-badge">🟢 Hand detector ready ({detector_type} API)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-badge" style="background:#fff0f0;color:#c0392b;">🔴 Hand detector unavailable</div>', unsafe_allow_html=True)
    if model:
        st.markdown(f'<div class="info-badge">🧠 Model loaded ({model.output_shape[-1]} classes)</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── Main columns ────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload ASL Image")
    
    uploaded = st.file_uploader(
        "Choose an image of your ASL hand sign",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    image = None
    img_np = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image, dtype=np.uint8)
        st.image(image, caption="Your uploaded image", use_container_width=True)

    st.markdown('<div class="tip-box">💡 <b>Tips for best results:</b><br>• Use good lighting<br>• Plain light background<br>• Hand clearly visible and centered<br>• Capture a single static ASL letter pose</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Prediction ──────────────────────────────────────────────────────────────
pred_char = "—"
confidence = 0.0
top3 = []
hand_found = False
annotated = None

if image is not None and model is not None and img_np is not None:
    input_shape = model.input_shape
    
    if len(input_shape) >= 2 and input_shape[1] == 63:
        # Landmark-based model
        coords, raw_lms = [], None
        
        if detector is not None:
            # Try 1: full image
            coords, raw_lms = detect_landmarks(detector_type, detector, img_np)
            
            # Try 2: center-cropped square
            if not coords:
                h, w = img_np.shape[:2]
                side = min(h, w)
                top = (h - side) // 2
                left_c = (w - side) // 2
                crop = img_np[top:top+side, left_c:left_c+side]
                coords, raw_lms = detect_landmarks(detector_type, detector, crop)
                if coords:
                    img_np = crop

        if len(coords) == 63:
            hand_found = True
            pred_char, confidence, top3 = predict(model, coords)
            try:
                annotated = annotate_hand(img_np, raw_lms)
            except Exception:
                annotated = None
    else:
        # Image-based model fallback
        hand_found = True
        ih = input_shape[1] or 64
        iw = input_shape[2] or 64
        ch = input_shape[3] if len(input_shape) > 3 else 3
        img_r = image.resize((iw, ih), Image.Resampling.LANCZOS)
        if ch == 1:
            img_r = img_r.convert("L")
            arr = np.expand_dims(np.array(img_r), -1)
        else:
            arr = np.array(img_r)
        arr = arr.astype("float32") / 255.0
        arr = np.expand_dims(arr, 0)
        preds = model.predict(arr, verbose=0)[0]
        nc = model.output_shape[-1]
        labels = LABELS_29 if nc == 29 else LABELS_26 if nc == 26 else [chr(ord('A') + i) for i in range(nc)]
        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [(labels[i] if i < len(labels) else str(i), float(preds[i] * 100)) for i in top3_idx]
        pred_char = top3[0][0]
        confidence = top3[0][1]


# ─── Right Panel ─────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✨ Prediction")

    if image is None:
        st.markdown("""
        <div class="result-box">
            <div class="result-letter" style="opacity:0.3;">🤟</div>
            <div class="result-word" style="color:#aaa;">Upload an image to begin</div>
        </div>
        """, unsafe_allow_html=True)

    elif not hand_found:
        st.markdown("""
        <div class="result-box" style="background:linear-gradient(135deg,#fff5f5,#fff0f0); border-color:#ffb3b3;">
            <div class="result-letter">❌</div>
            <div class="result-word" style="color:#c0392b;">No Hand Detected</div>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ MediaPipe could not find a hand in this image. Try: better lighting, plain background, hand filling the frame.")

    else:
        label_name = {
            'Delete': 'Delete', 'Nothing': 'Nothing', 'Space': 'Space'
        }.get(pred_char, f'Letter "{pred_char}"')
        
        st.markdown(f"""
        <div class="result-box">
            <div class="result-letter">{pred_char}</div>
            <div class="result-word">{label_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Confidence Breakdown (Top 3):**")
        for lbl, score in top3:
            st.markdown(f"**{lbl}** — {score:.1f}%")
            st.progress(min(1.0, score / 100.0))

    if annotated is not None:
        st.markdown("**Hand Landmarks Detected:**")
        st.image(annotated, caption="MediaPipe skeleton overlay", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center; color:#8c97b0; font-family:\'Fredoka\',sans-serif; font-size:1rem; padding-bottom:1rem;">✨ Small signs. Big conversations. ✨</div>', unsafe_allow_html=True)
