import cv2
import numpy as np
import os
import tensorflow as tf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def get_hand_detector():
    """Initializes MediaPipe HandLandmarker using tasks API or legacy mp.solutions."""
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
                min_hand_detection_confidence=0.15,
                min_hand_presence_confidence=0.15,
                min_tracking_confidence=0.15
            )
            return "tasks", vision.HandLandmarker.create_from_options(opts)
    except Exception:
        pass

    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            return "solutions", mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.2, min_tracking_confidence=0.2
            )
    except Exception:
        pass

    return None, None

def main():
    model_path = "best_cnn_asl_model.keras"
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_shape = model.input_shape
    print(f"Model loaded successfully! Expected input shape: {input_shape}")
    is_landmark_model = len(input_shape) >= 2 and input_shape[1] == 63

    num_classes = model.output_shape[-1]
    if num_classes == 29:
        labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['Delete', 'Nothing', 'Space']
    elif num_classes == 26:
        labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    else:
        labels = [chr(i) for i in range(ord('A'), ord('A') + num_classes)]

    detector_type, detector = get_hand_detector()
    if is_landmark_model and detector is None:
        print("Warning: Hand detector could not be initialized.")

    # Try DirectShow first on Windows to avoid MSMF errors
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n[!] Error: Could not access the webcam.")
        print("    If Streamlit or your browser is currently using the camera,")
        print("    please switch Streamlit to 'Upload Image' or close the camera tab first.\n")
        return

    print("\nStarting Real-Time ASL Recognition...")
    print("Hold your hand inside the green box.")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame. Exiting...")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        box_size = 280
        x1 = max(0, w - box_size - 40)
        y1 = 40
        x2 = min(w, x1 + box_size)
        y2 = min(h, y1 + box_size)

        roi = frame[y1:y2, x1:x2]
        pred_char = "—"
        confidence = 0.0
        hand_found = False

        if roi.size != 0:
            if is_landmark_model and detector is not None:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                landmarks = []
                pts = []

                if detector_type == "tasks":
                    import mediapipe as mp
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
                    res = detector.detect(mp_img)
                    if res.hand_landmarks and len(res.hand_landmarks) > 0:
                        hand_found = True
                        rh, rw, _ = roi.shape
                        for lm in res.hand_landmarks[0]:
                            landmarks.extend([lm.x, lm.y, lm.z])
                            pts.append((int(lm.x * rw), int(lm.y * rh)))
                elif detector_type == "solutions":
                    roi_copy = roi_rgb.copy()
                    roi_copy.flags.writeable = False
                    res = detector.process(roi_copy)
                    if res.multi_hand_landmarks and len(res.multi_hand_landmarks) > 0:
                        hand_found = True
                        rh, rw, _ = roi.shape
                        for lm in res.multi_hand_landmarks[0].landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                            pts.append((int(lm.x * rw), int(lm.y * rh)))

                if hand_found:
                    # Draw connections
                    for p1, p2 in CONNECTIONS:
                        if 0 <= p1 < len(pts) and 0 <= p2 < len(pts):
                            cv2.line(roi, pts[p1], pts[p2], (0, 255, 0), 2)
                    # Draw joints
                    for x, y in pts:
                        cv2.circle(roi, (x, y), 4, (0, 0, 255), -1)

                if len(landmarks) == 63:
                    inp_data = np.array(landmarks, dtype=np.float32).reshape(1, 63, 1)
                    predictions = model.predict(inp_data, verbose=0)[0]
                    pred_idx = int(np.argmax(predictions))
                    confidence = float(predictions[pred_idx] * 100)
                    pred_char = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)

            # Display prediction banner
            if hand_found and pred_char != "—":
                label_text = f"Sign: {pred_char} ({confidence:.1f}%)"
                cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 200, 0), cv2.FILLED)
                cv2.putText(frame, label_text, (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (80, 80, 80), cv2.FILLED)
                cv2.putText(frame, "No hand detected", (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        box_color = (0, 255, 0) if hand_found else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, "Place hand here", (x1 + 10, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1)

        cv2.putText(frame, "ASL Live Recognition", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Real-Time ASL Alphabet Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
