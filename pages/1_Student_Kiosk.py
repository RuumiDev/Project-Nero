import datetime
import math
import os
import pickle
import sqlite3
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from deepface import DeepFace


DB_PATH = "nero_attendance.db"
MATCH_THRESHOLD = 0.4
PROCESS_EVERY_N_FRAMES = 5


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


st.set_page_config(layout="wide")
st.title("Live Exam Attendance Kiosk")


def get_connection():
	return sqlite3.connect(DB_PATH)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
	a_norm = a / (np.linalg.norm(a) + 1e-10)
	b_norm = b / (np.linalg.norm(b) + 1e-10)
	return 1.0 - float(np.dot(a_norm, b_norm))


def load_enrolled_students():
	"""Load only students with saved encodings and build lookup dictionaries."""
	student_encodings = {}
	student_names = {}
	conn = None
	try:
		conn = get_connection()
		rows = conn.execute(
			"""
			SELECT student_id, name, face_encoding
			FROM students
			WHERE face_encoding IS NOT NULL
			"""
		).fetchall()
		for student_id, name, blob in rows:
			try:
				enc = np.array(pickle.loads(blob), dtype=np.float64)
				student_encodings[str(student_id)] = enc
				student_names[str(student_id)] = str(name) if name is not None else str(student_id)
			except Exception:
				continue
	finally:
		if conn is not None:
			conn.close()
	return student_encodings, student_names


def get_today_profile(student_id: str, today_str: str):
	"""Return student profile + today's seating assignment.

	Requested shape uses attendance.exam_date; this schema stores exam_date in exam_sessions,
	so this joins exam_sessions to filter by today's date.
	"""
	conn = None
	try:
		conn = get_connection()
		row = conn.execute(
			"""
			SELECT
				s.name,
				s.student_id,
				s.photo_data,
				a.exam_code,
				a.table_number,
				a.id
			FROM students s
			JOIN attendance_roster a
				ON s.student_id = a.student_id
			JOIN exam_sessions es
				ON es.exam_code = a.exam_code
			WHERE s.student_id = ?
				AND date(es.exam_date) = date(?)
			ORDER BY a.id DESC
			LIMIT 1
			""",
			(student_id, today_str),
		).fetchone()
		if row is None:
			return None

		# Explicit tuple extraction to avoid field misalignment issues.
		name, found_student_id, photo_data, exam_code, table_number, roster_id = row
		return {
			"name": name,
			"student_id": found_student_id,
			"photo_data": photo_data,
			"exam_code": exam_code,
			"table_number": table_number,
			"roster_id": roster_id,
		}
	finally:
		if conn is not None:
			conn.close()


def mark_present_by_roster_id(roster_id: int):
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	conn = None
	try:
		conn = get_connection()
		conn.execute(
			"""
			UPDATE attendance_roster
			SET status = 'Present', timestamp = ?
			WHERE id = ?
			""",
			(timestamp, roster_id),
		)
		conn.commit()
	finally:
		if conn is not None:
			conn.close()


def mark_present_for_today(student_id: str):
	today_str = datetime.date.today().strftime("%Y-%m-%d")
	profile = get_today_profile(student_id, today_str)
	if profile is None:
		return None, None
	mark_present_by_roster_id(profile["roster_id"])
	return profile["table_number"], profile["name"]


def get_today_exam_codes():
	"""Return sorted unique exam codes scheduled for today."""
	today = datetime.date.today().isoformat()
	conn = None
	try:
		conn = get_connection()
		rows = conn.execute(
			"""
			SELECT DISTINCT exam_code
			FROM exam_sessions
			WHERE exam_code IS NOT NULL
				AND TRIM(exam_code) != ''
				AND date(exam_date) = date(?)
			ORDER BY exam_code
			""",
			(today,),
		).fetchall()
		return [str(r[0]) for r in rows if r[0] is not None]
	finally:
		if conn is not None:
			conn.close()


def safe_release(cap):
	if cap is None:
		return
	try:
		cap.release()
	except Exception:
		pass


def draw_label(frame, text, x, y, color, text_scale, thickness):
	(tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
	pad = 8
	bg_x1 = max(0, x)
	bg_y1 = max(0, y - th - baseline - (pad * 2))
	bg_x2 = min(frame.shape[1] - 1, x + tw + (pad * 2))
	bg_y2 = max(0, y)
	if bg_y2 > bg_y1 and bg_x2 > bg_x1:
		cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)
		cv2.putText(
			frame,
			text,
			(bg_x1 + pad, bg_y2 - baseline - pad),
			cv2.FONT_HERSHEY_SIMPLEX,
			text_scale,
			(255, 255, 255),
			thickness,
			cv2.LINE_AA,
		)


def calculate_ear(eye_landmarks):
	"""Calculate eye aspect ratio using 6 points around an eye."""
	p1, p2, p3, p4, p5, p6 = eye_landmarks
	vertical_1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
	vertical_2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
	horizontal = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
	if horizontal <= 1e-6:
		return 0.0
	return (vertical_1 + vertical_2) / (2.0 * horizontal)


def ensure_face_landmarker_model():
	model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".model_cache"))
	model_path = os.path.join(model_dir, "face_landmarker.task")
	if os.path.exists(model_path):
		return model_path
	os.makedirs(model_dir, exist_ok=True)
	model_url = (
		"https://storage.googleapis.com/mediapipe-models/"
		"face_landmarker/face_landmarker/float16/1/face_landmarker.task"
	)
	try:
		urllib.request.urlretrieve(model_url, model_path)
		return model_path
	except Exception:
		return None


mp_face_mesh = None
mp_face_landmarker = None
LIVENESS_MODE = "none"

if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
	mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
	LIVENESS_MODE = "solutions"
else:
	try:
		from mediapipe.tasks.python.core.base_options import BaseOptions
		from mediapipe.tasks.python import vision
	except Exception:
		pass
	else:
		model_path = ensure_face_landmarker_model()
		if model_path is not None:
			try:
				options = vision.FaceLandmarkerOptions(
					base_options=BaseOptions(model_asset_path=model_path),
					running_mode=vision.RunningMode.VIDEO,
					num_faces=1,
					min_face_detection_confidence=0.5,
					min_face_presence_confidence=0.5,
					min_tracking_confidence=0.5,
				)
				mp_face_landmarker = vision.FaceLandmarker.create_from_options(options)
				LIVENESS_MODE = "tasks"
			except Exception:
				mp_face_landmarker = None

LIVENESS_ENABLED = LIVENESS_MODE != "none"


student_encoding_map, student_name_map = load_enrolled_students()
today_exam_codes = get_today_exam_codes()

with st.sidebar:
	activate_kiosk = st.checkbox("Activate Kiosk Camera")

if not student_encoding_map:
	st.warning("No enrolled students with face encodings found.")

if today_exam_codes:
	st.info(f"Today's Exams: {', '.join(today_exam_codes)}")
else:
	st.warning("No exam session is scheduled for today.")

video_placeholder = st.empty()
status_placeholder = st.empty()

if activate_kiosk:
	if not LIVENESS_ENABLED:
		status_placeholder.warning(
			"MediaPipe Face Mesh liveness is unavailable in this mediapipe build. "
			"Continuing with recognition-only mode."
		)
	elif LIVENESS_MODE == "tasks":
		status_placeholder.info("Liveness is running with MediaPipe Tasks fallback mode.")

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		status_placeholder.error("Could not open camera.")
		safe_release(cap)
	else:
		today_str = datetime.date.today().strftime("%Y-%m-%d")
		display_success_until = 0
		liveness_verified = not LIVENESS_ENABLED
		frame_idx = 0
		last_detections = []

		try:
			while cap.isOpened():
				ret, frame = cap.read()
				if not ret:
					time.sleep(0.01)
					continue

				# Keep camera buffer moving, but freeze UI/AI updates while profile pop-up is visible.
				if time.time() < display_success_until:
					continue

				mesh_faces = []
				if LIVENESS_MODE == "solutions" and mp_face_mesh is not None:
					rgb_for_mesh = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					mesh_results = mp_face_mesh.process(rgb_for_mesh)
					mesh_faces = mesh_results.multi_face_landmarks or []
				elif LIVENESS_MODE == "tasks" and mp_face_landmarker is not None:
					try:
						rgb_for_mesh = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_for_mesh)
						landmarker_result = mp_face_landmarker.detect_for_video(
							mp_image,
							int(time.time() * 1000),
						)
						mesh_faces = landmarker_result.face_landmarks or []
					except Exception:
						mesh_faces = []

				if not liveness_verified:
					pending_detections = []
					for face_landmarks in mesh_faces:
						coords = []
						iter_landmarks = face_landmarks.landmark if LIVENESS_MODE == "solutions" else face_landmarks
						for lm in iter_landmarks:
							x = int(lm.x * frame.shape[1])
							y = int(lm.y * frame.shape[0])
							coords.append((x, y))

						if len(coords) > max(max(LEFT_EYE_IDX), max(RIGHT_EYE_IDX)):
							left_eye = [coords[i] for i in LEFT_EYE_IDX]
							right_eye = [coords[i] for i in RIGHT_EYE_IDX]
							avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
							if avg_ear < 0.20:
								liveness_verified = True

						x_values = [p[0] for p in coords]
						y_values = [p[1] for p in coords]
						x1 = max(0, min(x_values))
						y1 = max(0, min(y_values))
						x2 = min(frame.shape[1] - 1, max(x_values))
						y2 = min(frame.shape[0] - 1, max(y_values))
						pending_detections.append((x1, y1, x2, y2))

					if not liveness_verified:
						for x1, y1, x2, y2 in pending_detections:
							cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
							draw_label(frame, "⚠️ PLEASE BLINK TO VERIFY", x1, max(20, y1 - 10), (0, 190, 190), 0.8, 2)

						clock_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
						(clock_w, clock_h), clock_baseline = cv2.getTextSize(
							clock_text,
							cv2.FONT_HERSHEY_SIMPLEX,
							0.75,
							2,
						)
						clock_pad = 8
						clock_x1 = max(0, frame.shape[1] - clock_w - (clock_pad * 2) - 12)
						clock_y1 = 12
						clock_x2 = frame.shape[1] - 12
						clock_y2 = clock_y1 + clock_h + clock_baseline + (clock_pad * 2)
						cv2.rectangle(frame, (clock_x1, clock_y1), (clock_x2, clock_y2), (20, 20, 20), cv2.FILLED)
						cv2.putText(
							frame,
							clock_text,
							(clock_x1 + clock_pad, clock_y2 - clock_baseline - clock_pad),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.75,
							(255, 255, 255),
							2,
							cv2.LINE_AA,
						)

						rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
						video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
						continue

				frame_idx += 1

				if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
					detections = []
					showed_profile_popup = False
					try:
						reps = DeepFace.represent(
							img_path=frame,
							model_name="Facenet",
							detector_backend="opencv",
							enforce_detection=True,
						)
						if isinstance(reps, dict):
							reps = [reps]

						for rep in reps:
							live_enc = np.array(rep["embedding"], dtype=np.float64)
							fa = rep["facial_area"]
							x1 = int(fa["x"])
							y1 = int(fa["y"])
							x2 = int(fa["x"] + fa["w"])
							y2 = int(fa["y"] + fa["h"])

							best_student_id = None
							best_dist = float("inf")
							for student_id, known_enc in student_encoding_map.items():
								dist = cosine_distance(live_enc, known_enc)
								if dist < best_dist:
									best_dist = dist
									best_student_id = student_id

							if best_student_id is not None and best_dist < MATCH_THRESHOLD:
								name = student_name_map.get(best_student_id, best_student_id)
								profile = get_today_profile(best_student_id, today_str)

								if profile is not None:
									mark_present_by_roster_id(profile["roster_id"])

									name = profile["name"]
									student_id = profile["student_id"]
									photo_data = profile["photo_data"]
									exam_code = profile["exam_code"]
									table_number = profile["table_number"]

									video_placeholder.empty()
									with video_placeholder.container():
										spacer1, main_col, spacer2 = st.columns([1, 2, 1])
										with main_col:
											with st.container(border=True):
												st.markdown(
													"<h3 style='text-align: center; color: #2e7d32; letter-spacing: 2px; margin-bottom: 0px;'>ATTENDANCE RECORDED</h3>",
													unsafe_allow_html=True,
												)
												st.divider()
												img_col, text_col = st.columns([1, 1.5])
												with img_col:
													if photo_data:
														st.image(photo_data, use_container_width=True)
													else:
														st.info("No stored profile photo.")
												with text_col:
													st.subheader(name)
													st.caption(f"**ID:** {student_id}  |  **Course:** {exam_code}")
													st.markdown(
														f"<h1 style='color: #4CAF50; margin-top: 15px; font-size: 3.5rem;'>TABLE {table_number}</h1>",
														unsafe_allow_html=True,
													)

									display_success_until = time.time() + 5
									liveness_verified = not LIVENESS_ENABLED
									showed_profile_popup = True
									break

								detections.append(
									{
										"box": (x1, y1, x2, y2),
										"recognized": True,
										"name": name,
										"table": None,
									}
								)
							else:
								detections.append(
									{
										"box": (x1, y1, x2, y2),
										"recognized": False,
										"name": "UNKNOWN",
										"table": None,
									}
								)
					except ValueError:
						detections = []

					if showed_profile_popup:
						continue

					last_detections = detections

				for det in last_detections:
					x1, y1, x2, y2 = det["box"]
					if det["recognized"]:
						cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
						name_text = det["name"]
						table_text = f"TABLE: {det['table']}" if det["table"] is not None else "TABLE: TODAY N/A"
						draw_label(frame, table_text, x1, max(20, y1 - 48), (0, 160, 0), 1.1, 3)
						draw_label(frame, name_text, x1, max(20, y1 - 10), (0, 120, 0), 0.8, 2)
					else:
						cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
						draw_label(frame, "UNKNOWN", x1, max(20, y1 - 10), (0, 0, 180), 0.8, 2)

				# Live timestamp overlay for invigilator confidence and audit visibility.
				clock_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				(clock_w, clock_h), clock_baseline = cv2.getTextSize(
					clock_text,
					cv2.FONT_HERSHEY_SIMPLEX,
					0.75,
					2,
				)
				clock_pad = 8
				clock_x1 = max(0, frame.shape[1] - clock_w - (clock_pad * 2) - 12)
				clock_y1 = 12
				clock_x2 = frame.shape[1] - 12
				clock_y2 = clock_y1 + clock_h + clock_baseline + (clock_pad * 2)
				cv2.rectangle(frame, (clock_x1, clock_y1), (clock_x2, clock_y2), (20, 20, 20), cv2.FILLED)
				cv2.putText(
					frame,
					clock_text,
					(clock_x1 + clock_pad, clock_y2 - clock_baseline - clock_pad),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.75,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)

				rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
		finally:
			safe_release(cap)
			if mp_face_mesh is not None:
				mp_face_mesh.close()
			if mp_face_landmarker is not None:
				mp_face_landmarker.close()

with st.expander("Manual Override (Invigilator Access)"):
	manual_student_id = st.text_input("Student ID")
	if st.button("Mark Present"):
		sid = manual_student_id.strip()
		if not sid:
			st.error("Please enter a Student ID.")
		else:
			table_no, db_name = mark_present_for_today(sid)
			if table_no is None:
				st.error("No roster entry found for this student in today's exam.")
			else:
				display_name = db_name if db_name else sid
				st.success(f"Marked Present: {display_name} ({sid}) | TABLE: {table_no}")
