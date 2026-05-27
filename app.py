import streamlit as st
import sqlite3
import cv2
import numpy as np
import pickle
import time
import os
import urllib.request
import math
import pandas as pd
from PIL import Image
from deepface import DeepFace

# ── Constants ─────────────────────────────────────────────────────────────────

THRESHOLD   = 0.4
SKIP_FRAMES = 5
CAMERA_INDEXES = [0, 1, 2, 3]

# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect("nero.db")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY,
            name          TEXT,
            role          TEXT,
            face_encoding BLOB
        )
        """
    )
    con.commit()
    con.close()

init_db()


def load_users():
    con = sqlite3.connect("nero.db")
    rows = con.execute("SELECT name, role, face_encoding FROM users").fetchall()
    con.close()
    return [(n, r, pickle.loads(b)) for n, r, b in rows if b]


def get_all_users():
    con = sqlite3.connect("nero.db")
    rows = con.execute("SELECT id, name, role FROM users ORDER BY id").fetchall()
    con.close()
    return rows


def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return 1.0 - float(np.dot(a, b))

# ── Camera helpers ────────────────────────────────────────────────────────────

def _safe_release(cap):
    if cap is None:
        return
    try:
        cap.release()
    except Exception:
        pass

def _open_camera_with_fallback():
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MSMF"),
        (None, "Default"),
    ]

    for cam_index in CAMERA_INDEXES:
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(cam_index, backend) if backend is not None else cv2.VideoCapture(cam_index)
            except cv2.error:
                continue

            if cap is None or not cap.isOpened():
                if cap is not None:
                    _safe_release(cap)
                continue

            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                _safe_release(cap)
                continue

            # Validate by reading a few warm-up frames; some backends open but never deliver.
            got_frame = False
            for _ in range(12):
                try:
                    ret, _ = cap.read()
                except cv2.error:
                    ret = False
                if ret:
                    got_frame = True
                    break
                time.sleep(0.02)

            if got_frame:
                return cap, f"{name} (camera {cam_index})"

            _safe_release(cap)

    return None, None

def run_camera_preview():
    cap, backend_name = _open_camera_with_fallback()
    if cap is None:
        st.error("Unable to get frames from any camera index (0-3). Close camera apps and retry.")
        return

    st.caption(f"Camera backend: {backend_name}")
    placeholder = st.empty()
    try:
        while True:
            try:
                ret, frame = cap.read()
            except cv2.error:
                st.warning("Camera backend raised an OpenCV error. Trying to continue...")
                time.sleep(0.05)
                continue
            if not ret:
                time.sleep(0.01)
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except cv2.error:
                time.sleep(0.01)
                continue
            placeholder.image(rgb, channels="RGB", use_container_width=True)
    finally:
        _safe_release(cap)


def run_face_recognition(users):
    cap, backend_name = _open_camera_with_fallback()
    if cap is None:
        st.error("Unable to get frames from any camera index (0-3). Close camera apps and retry.")
        return

    st.caption(f"Camera backend: {backend_name}")
    placeholder = st.empty()
    last_boxes = []
    frame_idx = 0

    try:
        while True:
            try:
                ret, frame = cap.read()
            except cv2.error:
                time.sleep(0.05)
                continue
            if not ret:
                time.sleep(0.01)
                continue

            frame_idx += 1

            if frame_idx % SKIP_FRAMES == 0:
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                try:
                    reps = DeepFace.represent(
                        img_path=small_rgb,
                        model_name="Facenet",
                        detector_backend="opencv",
                        enforce_detection=True,
                    )
                    last_boxes = []
                    for rep in reps:
                        enc = np.array(rep["embedding"], dtype=np.float64)
                        fa = rep["facial_area"]
                        x1, y1 = int(fa["x"] * 2), int(fa["y"] * 2)
                        x2, y2 = int(x1 + fa["w"] * 2), int(y1 + fa["h"] * 2)

                        best_name, best_role, best_dist = "Unknown", "", float("inf")
                        for name, role, stored in users:
                            dist = cosine_distance(enc, stored)
                            if dist < best_dist:
                                best_dist, best_name, best_role = dist, name, role

                        if best_dist <= THRESHOLD:
                            label, color = f"{best_name} [{best_role}]", (0, 200, 0)
                        else:
                            label, color = "Unknown", (0, 60, 255)

                        last_boxes.append((x1, y1, x2, y2, label, color))
                except ValueError:
                    last_boxes = []

            annotated = frame.copy()
            for (x1, y1, x2, y2, label, color) in last_boxes:
                # Main face box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.65
                text_thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                pad_x = 6
                pad_y = 4
                box_h = text_h + baseline + (pad_y * 2)

                # Prefer label bar above face box; if too close to top, place below.
                bg_x1 = max(0, x1)
                bg_x2 = min(annotated.shape[1] - 1, x1 + text_w + (pad_x * 2))
                preferred_bg_y2 = y1 - 2
                if preferred_bg_y2 - box_h >= 0:
                    bg_y1 = preferred_bg_y2 - box_h
                    bg_y2 = preferred_bg_y2
                else:
                    bg_y1 = min(annotated.shape[0] - 1, y1 + 2)
                    bg_y2 = min(annotated.shape[0] - 1, bg_y1 + box_h)

                cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)

                # Use high-contrast text color for readability against the filled label bar.
                b, g, r = color
                luminance = (0.114 * b) + (0.587 * g) + (0.299 * r)
                text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)

                cv2.putText(
                    annotated,
                    label,
                    (bg_x1 + pad_x, bg_y2 - baseline - pad_y),
                    font,
                    font_scale,
                    text_color,
                    text_thickness,
                    cv2.LINE_AA,
                )

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb, channels="RGB", use_container_width=True)
    finally:
        _safe_release(cap)


def run_hand_tracking():
    try:
        import mediapipe as mp
    except ModuleNotFoundError:
        st.error(
            "mediapipe is not installed in the Python environment running Streamlit. "
            "Run: python -m pip install mediapipe"
        )
        return

    def ensure_hand_model():
        model_dir = os.path.join(os.path.dirname(__file__), ".model_cache")
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        if os.path.exists(model_path):
            return model_path
        os.makedirs(model_dir, exist_ok=True)
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        try:
            urllib.request.urlretrieve(model_url, model_path)
            return model_path
        except Exception:
            return None

    cap, backend_name = _open_camera_with_fallback()
    if cap is None:
        st.error("Unable to get frames from any camera index (0-3). Close camera apps and retry.")
        return

    st.caption(f"Camera backend: {backend_name}")
    placeholder = st.empty()
    status_box = st.empty()

    def count_fingers(landmarks, hand_label):
        total = 0

        # Thumb uses distance from pinky MCP base to avoid handedness/mirroring issues.
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        pinky_base = landmarks[17]

        tip_to_pinky = math.hypot(thumb_tip.x - pinky_base.x, thumb_tip.y - pinky_base.y)
        ip_to_pinky = math.hypot(thumb_ip.x - pinky_base.x, thumb_ip.y - pinky_base.y)
        if tip_to_pinky > ip_to_pinky:
            total += 1

        # Index, Middle, Ring, Pinky are up when tip is above PIP (smaller y).
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip_idx, pip_idx in finger_pairs:
            if landmarks[tip_idx].y < landmarks[pip_idx].y:
                total += 1

        return total

    # Prefer classic Solutions API when available; fallback to Tasks API for builds without mp.solutions.
    if hasattr(mp, "solutions"):
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            try:
                while True:
                    try:
                        ret, frame = cap.read()
                    except cv2.error:
                        time.sleep(0.05)
                        continue

                    if not ret:
                        time.sleep(0.01)
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)

                    total_fingers = 0

                    if results.multi_hand_landmarks:
                        handedness_list = results.multi_handedness or []
                        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            mp_draw.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                            )

                            lm = hand_landmarks.landmark
                            hand_label = None
                            if idx < len(handedness_list) and handedness_list[idx].classification:
                                hand_label = handedness_list[idx].classification[0].label
                            total_fingers += count_fingers(lm, hand_label)

                    cv2.putText(
                        frame,
                        f"Total Fingers: {total_fingers}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )

                    if total_fingers == 5:
                        status_box.success("Gesture Recognized!")
                    else:
                        status_box.empty()

                    display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    placeholder.image(display, channels="RGB", use_container_width=True)
            finally:
                _safe_release(cap)
        return

    # Tasks fallback for minimal builds that expose only mediapipe.tasks
    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python import vision
    except Exception:
        _safe_release(cap)
        st.error("This mediapipe build does not include hand-tracking APIs required for this feature.")
        return

    model_path = ensure_hand_model()
    if model_path is None:
        _safe_release(cap)
        st.error("Could not download hand landmark model. Check internet connection and try again.")
        return

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    try:
        while True:
            try:
                ret, frame = cap.read()
            except cv2.error:
                time.sleep(0.05)
                continue

            if not ret:
                time.sleep(0.01)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect_for_video(mp_image, int(time.time() * 1000))

            total_fingers = 0
            if result.hand_landmarks:
                h, w, _ = frame.shape
                handedness_list = result.handedness or []
                for idx, landmarks in enumerate(result.hand_landmarks):
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

                    for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 255), 2)
                    for p in pts:
                        cv2.circle(frame, p, 3, (255, 0, 255), -1)

                    hand_label = None
                    if idx < len(handedness_list) and len(handedness_list[idx]) > 0:
                        cat = handedness_list[idx][0]
                        hand_label = (
                            getattr(cat, "category_name", None)
                            or getattr(cat, "display_name", None)
                            or getattr(cat, "label", None)
                        )

                    total_fingers += count_fingers(landmarks, hand_label)

            cv2.putText(
                frame,
                f"Total Fingers: {total_fingers}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

            if total_fingers == 5:
                status_box.success("Gesture Recognized!")
            else:
                status_box.empty()

            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(display, channels="RGB", use_container_width=True)
    finally:
        _safe_release(cap)
        if hasattr(detector, "close"):
            detector.close()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Project Nero", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Project Nero Settings")
    page = st.radio(
        "Menu",
        ["Dashboard", "Add New User", "Face Recognition", "Hand Tracking", "Emotion Detection"],
    )
    st.divider()
    camera_on = st.checkbox("Enable V10 Camera")

# ── Main content ──────────────────────────────────────────────────────────────

if page == "Dashboard":
    st.title("Welcome to Project Nero")
    st.markdown("---")

    all_users = get_all_users()
    c1, c2, c3 = st.columns(3)
    c1.metric("Registered Users", len(all_users))
    c2.metric("Active Sessions", "1")
    c3.metric("System Status", "Online")

    st.markdown("---")
    if all_users:
        st.subheader("Registered Users")
        df = pd.DataFrame(all_users, columns=["ID", "Name", "Role"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No users registered yet. Go to 'Add New User' to get started.")

    if camera_on:
        st.subheader("Live Camera Feed")
        run_camera_preview()

elif page == "Add New User":
    st.title("Add New User")

    full_name = st.text_input("Full Name")
    role = st.text_input("Role")
    source_mode = st.radio("Input Source", ["Use V10 Camera", "Upload Photo"], horizontal=True)

    uploaded_rgb = None
    if source_mode == "Upload Photo":
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                uploaded_rgb = np.array(Image.open(uploaded_file).convert("RGB"))
                st.image(uploaded_rgb, caption="Uploaded photo preview", use_container_width=True)
            except Exception:
                st.error("Could not read the uploaded image. Please upload a valid JPG/JPEG/PNG file.")

    process_btn = st.button("Process & Save")

    if process_btn:
        if not full_name.strip():
            st.warning("Please enter a name before capturing.")
        elif source_mode == "Upload Photo" and uploaded_rgb is None:
            st.warning("Please upload a valid photo first.")
        else:
            if source_mode == "Upload Photo":
                try:
                    results = DeepFace.represent(
                        img_path=uploaded_rgb,
                        model_name="Facenet",
                        detector_backend="opencv",
                        enforce_detection=True,
                    )
                    enc = np.array(results[0]["embedding"], dtype=np.float64)
                    blob = pickle.dumps(enc)

                    con = sqlite3.connect("nero.db")
                    con.execute(
                        "INSERT INTO users (name, role, face_encoding) VALUES (?, ?, ?)",
                        (full_name.strip(), role.strip(), blob),
                    )
                    con.commit()
                    con.close()

                    st.success(f"User '{full_name.strip()}' saved successfully from uploaded photo!")
                except ValueError:
                    st.warning("No face detected in uploaded photo. Please choose another image.")
            else:
                cap, backend_name = _open_camera_with_fallback()
                if cap is None:
                    st.error("Unable to open camera for capture.")
                else:
                    st.caption(f"Camera backend: {backend_name}")
                    progress_bar = st.progress(0, text="Starting multi-shot capture...")
                    encodings = []
                    last_rgb_frame = None
                    capture_failed = False

                    for i in range(3):
                        ok, frame = cap.read()
                        if not ok:
                            st.error("Camera opened but failed to capture a frame.")
                            capture_failed = True
                            break

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        last_rgb_frame = rgb

                        try:
                            results = DeepFace.represent(
                                img_path=rgb,
                                model_name="Facenet",
                                detector_backend="opencv",
                                enforce_detection=True,
                            )
                            enc = np.array(results[0]["embedding"], dtype=np.float64)
                            encodings.append(enc)
                        except ValueError:
                            st.warning(
                                f"No face detected in capture {i + 1}/3. Please try again with better lighting/positioning."
                            )
                            capture_failed = True
                            break

                        progress_bar.progress(
                            int(((i + 1) / 3) * 100),
                            text=f"Captured photo {i + 1}/3",
                        )

                        if i < 2:
                            time.sleep(1)

                    _safe_release(cap)

                    if not capture_failed and len(encodings) == 3:
                        avg_encoding = np.mean(np.stack(encodings, axis=0), axis=0)
                        blob = pickle.dumps(avg_encoding)

                        con = sqlite3.connect("nero.db")
                        con.execute(
                            "INSERT INTO users (name, role, face_encoding) VALUES (?, ?, ?)",
                            (full_name.strip(), role.strip(), blob),
                        )
                        con.commit()
                        con.close()

                        progress_bar.progress(100, text="Capture complete")
                        st.success(f"User '{full_name.strip()}' saved successfully with averaged encoding!")
                        if last_rgb_frame is not None:
                            st.image(last_rgb_frame, caption="Final captured photo", use_container_width=True)

    st.markdown("---")
    st.subheader("Registered Users")
    all_users = get_all_users()
    if all_users:
        df = pd.DataFrame(all_users, columns=["ID", "Name", "Role"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No users registered yet.")

    if camera_on and source_mode == "Use V10 Camera":
        st.subheader("Live Camera Preview")
        run_camera_preview()

elif page == "Face Recognition":
    st.title("Face Recognition")
    users = load_users()

    if not users:
        st.warning("No users enrolled yet. Go to 'Add New User' first.")
    elif not camera_on:
        st.info("Enable the V10 Camera from the sidebar to start live recognition.")
    else:
        run_face_recognition(users)

elif page == "Hand Tracking":
    st.title("Hand Tracking")
    st.info("Show your hand to track landmarks and count raised fingers.")
    if camera_on:
        run_hand_tracking()

elif page == "Emotion Detection":
    st.title("Emotion Detection")
    st.info("Emotion detection module coming soon.")
    if camera_on:
        run_camera_preview()
