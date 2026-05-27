import pandas as pd
import streamlit as st
import pickle
import io
import numpy as np
import sqlite3
import cv2
import time
from PIL import Image
from deepface import DeepFace

from database import get_connection, init_db


st.set_page_config(page_title="Admin Dashboard", layout="wide")


with st.sidebar:
	st.header("Admin Access")
	admin_password = st.text_input("Enter password", type="password")


if admin_password != "admin123":
	st.title("Admin Dashboard")
	st.warning("Please enter the correct admin password in the sidebar to access this page.")
	st.stop()


init_db()

st.title("Admin Dashboard")


def load_master_df():
	conn = get_connection()
	query = """
	WITH master AS (
		SELECT
			ar.id AS roster_row_id,
			s.id AS student_row_id,
			COALESCE(ar.student_id, s.student_id) AS student_id,
			s.name AS name,
			ar.exam_code AS exam_code,
			es.exam_date AS exam_date,
			ar.table_number AS table_number,
			ar.status AS status,
			ar.timestamp AS timestamp,
			(s.face_encoding IS NOT NULL) AS has_encoding
		FROM attendance_roster ar
		LEFT JOIN students s
			ON s.student_id = ar.student_id
		LEFT JOIN exam_sessions es
			ON es.exam_code = ar.exam_code

		UNION ALL

		SELECT
			NULL AS roster_row_id,
			s.id AS student_row_id,
			s.student_id AS student_id,
			s.name AS name,
			NULL AS exam_code,
			NULL AS exam_date,
			NULL AS table_number,
			NULL AS status,
			NULL AS timestamp,
			(s.face_encoding IS NOT NULL) AS has_encoding
		FROM students s
		LEFT JOIN attendance_roster ar
			ON ar.student_id = s.student_id
		WHERE ar.student_id IS NULL
	)
	SELECT * FROM master
	ORDER BY student_id, exam_code
	"""
	df = pd.read_sql_query(query, conn)
	conn.close()
	return df


master_df = load_master_df()

# Required display fields
master_df["Photo_Status"] = np.where(master_df["has_encoding"].astype(bool), "✅ Enrolled", "🛑 NO ID")
master_df["Exam Code"] = master_df["exam_code"].fillna("").astype(str).str.strip()
master_df.loc[master_df["Exam Code"] == "", "Exam Code"] = "⚠️ NO EXAM"

master_df["Exam Date"] = pd.to_datetime(master_df["exam_date"], errors="coerce")

master_df["Table Number"] = pd.to_numeric(master_df["table_number"], errors="coerce")

display_df = pd.DataFrame(
	{
		"Student ID": master_df["student_id"],
		"Name": master_df["name"],
		"Exam Code": master_df["Exam Code"],
		"Exam Date": master_df["Exam Date"],
		"Table Number": master_df["Table Number"],
		"Status": master_df["status"],
		"Timestamp": master_df["timestamp"],
		"Has_Encoding": master_df["has_encoding"].astype(bool),
		"Photo_Status": master_df["Photo_Status"],
		"_roster_row_id": master_df["roster_row_id"],
		"_student_row_id": master_df["student_row_id"],
		"_orig_exam": master_df["exam_code"],
		"_orig_exam_date": master_df["exam_date"],
	}
)

col1, col2 = st.columns([3, 1])


with col1:
	st.subheader("Master Grid")
	exam_options = sorted(display_df["Exam Code"].dropna().unique().tolist())
	selected_exams = st.multiselect("Filter by Exam Code", options=exam_options)
	exam_date_options = sorted(display_df["Exam Date"].dropna().unique().tolist())
	selected_exam_dates = st.multiselect("Filter by Exam Date", options=exam_date_options)

	filtered_df = display_df.copy()
	if selected_exams:
		filtered_df = filtered_df[filtered_df["Exam Code"].isin(selected_exams)]
	if selected_exam_dates:
		filtered_df = filtered_df[filtered_df["Exam Date"].isin(selected_exam_dates)]
	filtered_df = filtered_df.reset_index(drop=True)

	total_students = int(len(filtered_df))
	present_students = int(filtered_df["Status"].fillna("").astype(str).str.lower().eq("present").sum())

	metric_col1, metric_col2 = st.columns(2)
	metric_col1.metric("Total Students", total_students)
	metric_col2.metric("Present", present_students)

	st.data_editor(
		filtered_df,
		key="master_grid",
		num_rows="dynamic",
		use_container_width=True,
		column_config={
			"_roster_row_id": None,
			"_student_row_id": None,
			"_orig_exam": None,
			"_orig_exam_date": None,
			"Student ID": st.column_config.TextColumn("Student ID", required=True),
			"Name": st.column_config.TextColumn("Name"),
			"Exam Code": st.column_config.TextColumn("Exam Code"),
			"Exam Date": st.column_config.DateColumn("Exam Date"),
			"Table Number": st.column_config.NumberColumn("Table Number", step=1),
			"Status": st.column_config.TextColumn("Status", disabled=True),
			"Timestamp": st.column_config.TextColumn("Timestamp", disabled=True),
			"Photo_Status": st.column_config.TextColumn(disabled=True),
			"Has_Encoding": st.column_config.CheckboxColumn(disabled=True),
		},
	)

	export_df = filtered_df.drop(
		columns=[
			"Has_Encoding",
			"Photo_Status",
			"_roster_row_id",
			"_student_row_id",
			"_orig_exam",
			"_orig_exam_date",
		],
		errors="ignore",
	)
	csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")

	with st.container():
		export_left, export_right = st.columns([1, 3])
		with export_left:
			st.download_button(
				label="📄 Export Official Roster (.csv)",
				data=csv_bytes,
				file_name=f"Attendance_Report.csv",
				mime="text/csv",
			)

	grid_state = st.session_state.get("master_grid", {})
	edited_rows = grid_state.get("edited_rows", {})
	added_rows = grid_state.get("added_rows", [])
	deleted_rows = grid_state.get("deleted_rows", [])
	has_changes = (len(edited_rows) > 0) or (len(added_rows) > 0) or (len(deleted_rows) > 0)

	if has_changes:
		st.warning("⚠️ You have unsaved changes in the grid.")

	if has_changes and st.button("Save Grid Changes"):

		conn = None
		try:
			conn = get_connection()
			cursor = conn.cursor()

			def normalize_exam_code(v):
				if v is None:
					return None
				text = str(v).strip()
				return None if text in ("", "⚠️ NO EXAM") else text

			def normalize_exam_date(v):
				if v is None:
					return None
				if isinstance(v, pd.Timestamp):
					return v.date().isoformat()
				text = str(v).strip()
				return None if text in ("", "⚠️ NO DATE") else text

			def normalize_table(v):
				if v is None:
					return None
				if isinstance(v, str) and v.strip() in ("", "⚠️ NO TABLE"):
					return None
				try:
					return int(v)
				except (TypeError, ValueError):
					return None

			# Added rows.
			for row in added_rows:
				student_id = str(row.get("Student ID", "")).strip()
				name = str(row.get("Name", "")).strip()
				exam_code = normalize_exam_code(row.get("Exam Code"))
				exam_date = normalize_exam_date(row.get("Exam Date"))
				table_number = normalize_table(row.get("Table Number"))

				if not student_id:
					continue

				cursor.execute(
					"INSERT OR IGNORE INTO students (student_id, name) VALUES (?, ?)",
					(student_id, name if name else None),
				)
				if name:
					cursor.execute(
						"UPDATE students SET name = ? WHERE student_id = ?",
						(name, student_id),
					)

				cursor.execute(
					"""
					INSERT INTO attendance_roster (student_id, exam_code, table_number)
					VALUES (?, ?, ?)
					""",
					(student_id, exam_code, table_number),
				)

				if exam_code is not None:
					cursor.execute(
						"""
						INSERT INTO exam_sessions (exam_code, subject, exam_date, exam_time)
						VALUES (?, NULL, ?, NULL)
						ON CONFLICT(exam_code) DO UPDATE SET exam_date = excluded.exam_date
						""",
						(exam_code, exam_date),
					)

			# Edited rows (row index maps to original filtered dataframe).
			for row_idx, changes in edited_rows.items():
				idx = int(row_idx)
				if idx < 0 or idx >= len(filtered_df):
					continue

				base_row = filtered_df.iloc[idx]
				student_id = str(base_row.get("Student ID", "")).strip()
				if not student_id:
					continue

				if "Name" in changes:
					new_name = str(changes["Name"]).strip()
					existing_student = cursor.execute(
						"SELECT 1 FROM students WHERE student_id = ? LIMIT 1",
						(student_id,),
					).fetchone()
					if existing_student:
						cursor.execute(
							"UPDATE students SET name = ? WHERE student_id = ?",
							(new_name if new_name else None, student_id),
						)
					else:
						cursor.execute(
							"INSERT INTO students (student_id, name) VALUES (?, ?)",
							(student_id, new_name if new_name else None),
						)

				exam_code = normalize_exam_code(base_row.get("_orig_exam"))
				exam_date = normalize_exam_date(base_row.get("_orig_exam_date"))
				table_number = normalize_table(base_row.get("Table Number"))

				if "Exam Code" in changes:
					exam_code = normalize_exam_code(changes["Exam Code"])
				if "Exam Date" in changes:
					exam_date = normalize_exam_date(changes["Exam Date"])
				if "Table Number" in changes:
					table_number = normalize_table(changes["Table Number"])

				if ("Exam Code" in changes) or ("Exam Date" in changes) or ("Table Number" in changes):
					existing_roster = cursor.execute(
						"SELECT id FROM attendance_roster WHERE student_id = ? LIMIT 1",
						(student_id,),
					).fetchone()

					if existing_roster:
						cursor.execute(
							"""
							UPDATE attendance_roster
							SET exam_code = ?, table_number = ?
							WHERE id = ?
							""",
							(exam_code, table_number, existing_roster[0]),
						)
					else:
						cursor.execute(
							"""
							INSERT INTO attendance_roster (student_id, exam_code, table_number)
							VALUES (?, ?, ?)
							""",
							(student_id, exam_code, table_number),
						)

					if exam_code is not None:
						cursor.execute(
							"""
							INSERT INTO exam_sessions (exam_code, subject, exam_date, exam_time)
							VALUES (?, NULL, ?, NULL)
							ON CONFLICT(exam_code) DO UPDATE SET exam_date = excluded.exam_date
							""",
							(exam_code, exam_date),
						)

			# Deleted rows.
			for row_idx in deleted_rows:
				idx = int(row_idx)
				if idx < 0 or idx >= len(filtered_df):
					continue
				student_id = str(filtered_df.iloc[idx].get("Student ID", "")).strip()
				if not student_id:
					continue
				cursor.execute("DELETE FROM attendance_roster WHERE student_id = ?", (student_id,))
				cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))

			conn.commit()
			st.toast("✅ Database updated successfully!", icon="🎉")
			st.success("Changes saved.")
			time.sleep(1)
			st.rerun()
		except Exception as e:
			st.error(f"❌ Failed to save: {e}")
		finally:
			if conn is not None:
				conn.close()


with col2:
	st.subheader("Action Panel")

	with st.expander("1. Mass Upload Roster (.xlsx)", expanded=True):
		st.info('Required Columns: Student ID, Exam Code, Table Number (Variations like "Seat" or "Course" are accepted).')
		roster_file = st.file_uploader("Upload roster file", type=["xlsx"], key="mass_roster_upload")

		if roster_file is not None:
			try:
				roster_df = pd.read_excel(roster_file)
				roster_df.columns = roster_df.columns.astype(str).str.lower().str.strip()
				column_map = {
					"id": "student_id",
					"matric no": "student_id",
					"student_id": "student_id",
					"course": "exam_code",
					"code": "exam_code",
					"exam_code": "exam_code",
					"table": "table_number",
					"seat": "table_number",
				}
				roster_df = roster_df.rename(columns=column_map)
				roster_df = roster_df.loc[:, ~roster_df.columns.duplicated()]
			except Exception as exc:
				st.error(f"Failed to read Excel file: {exc}")
			else:
				st.dataframe(roster_df.head(), use_container_width=True)

				if st.button("Save Roster to Database", key="save_mass_roster"):
					required_columns = ["student_id", "exam_code", "table_number"]
					missing_columns = [col for col in required_columns if col not in roster_df.columns]

					if missing_columns:
						st.error("Missing required columns: " + ", ".join(missing_columns))
					else:
						conn = None
						inserted_rows = 0
						try:
							conn = get_connection()
							cursor = conn.cursor()

							for _, row in roster_df.iterrows():
								student_id = str(row["student_id"]).strip() if pd.notna(row["student_id"]) else ""
								exam_code = str(row["exam_code"]).strip() if pd.notna(row["exam_code"]) else ""
								table_number = None
								if pd.notna(row["table_number"]):
									try:
										table_number = int(float(str(row["table_number"]).strip()))
									except (TypeError, ValueError):
										table_number = None

								if not student_id or not exam_code or table_number is None:
									continue

								cursor.execute(
									"""
									INSERT INTO attendance_roster (student_id, exam_code, table_number)
									VALUES (?, ?, ?)
									""",
									(student_id, exam_code, table_number),
								)
								inserted_rows += 1

							conn.commit()
							st.success(f"Roster saved successfully. Inserted {inserted_rows} records.")
						except Exception as exc:
							st.error(f"Failed to save roster: {exc}")
						finally:
							if conn is not None:
								conn.close()

	with st.expander("2. Resolve Missing IDs", expanded=True):
		missing_id_df = display_df[display_df["Photo_Status"] == "🛑 NO ID"]
		missing_ids = sorted(missing_id_df["Student ID"].dropna().astype(str).unique().tolist())

		selected_missing_id = st.selectbox(
			"Student ID with missing photo",
			options=missing_ids if missing_ids else [""],
		)
		missing_photo_file = st.file_uploader(
			"Upload student photo (JPG/PNG)",
			type=["jpg", "jpeg", "png"],
			key="resolve_missing_id_photo",
		)

		if missing_photo_file is not None:
			try:
				preview_img = Image.open(missing_photo_file).convert("RGB")
				st.image(preview_img, caption="Uploaded photo", width=220)
			except Exception as exc:
				st.error(f"Could not read uploaded photo: {exc}")

		if st.button("Update Missing Photo", key="update_missing_photo"):
			if not selected_missing_id:
				st.error("No missing Student ID selected.")
			elif missing_photo_file is None:
				st.error("Please upload a photo.")
			else:
				conn = None
				try:
					photo_bytes = missing_photo_file.getvalue()
					pil_img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
					rgb_img = np.array(pil_img)
					opencv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

					results = DeepFace.represent(
						img_path=opencv_img,
						model_name="Facenet",
						detector_backend="opencv",
						enforce_detection=True,
					)

					encoding = np.array(results[0]["embedding"], dtype=np.float64)
					encoding_blob = pickle.dumps(encoding)

					conn = get_connection()
					cursor = conn.cursor()

					row = cursor.execute(
						"SELECT id, name FROM students WHERE student_id = ?",
						(selected_missing_id,),
					).fetchone()

					if row:
						cursor.execute(
							"UPDATE students SET face_encoding = ?, photo_data = ? WHERE student_id = ?",
							(encoding_blob, photo_bytes, selected_missing_id),
						)
					else:
						name_from_grid = "Unknown"
						match = missing_id_df[missing_id_df["Student ID"].astype(str) == str(selected_missing_id)]
						if not match.empty:
							candidate_name = str(match.iloc[0]["Name"])
							if candidate_name and candidate_name.lower() != "nan":
								name_from_grid = candidate_name

						cursor.execute(
							"""
							INSERT INTO students (student_id, name, face_encoding, photo_data)
							VALUES (?, ?, ?, ?)
							""",
							(selected_missing_id, name_from_grid, encoding_blob, photo_bytes),
						)

					conn.commit()
					st.success(f"Photo resolved for Student ID {selected_missing_id}.")
				except ValueError:
					st.error("No face detected in uploaded photo. Please upload a clear face image.")
				except Exception as exc:
					st.error(f"Failed to update missing photo: {exc}")
				finally:
					if conn is not None:
						conn.close()

	with st.expander("Manage Photo ID", expanded=False):
		conn = None
		student_options = []
		try:
			conn = get_connection()
			students_df = pd.read_sql_query(
				"SELECT student_id, name FROM students ORDER BY student_id",
				conn,
			)
			for _, r in students_df.iterrows():
				sid = str(r["student_id"]).strip()
				name = "" if pd.isna(r["name"]) else str(r["name"]).strip()
				if sid:
					student_options.append((sid, name))
		except Exception as exc:
			st.error(f"Failed to load students: {exc}")
		finally:
			if conn is not None:
				conn.close()

		selected_student = st.selectbox(
			"Select Student",
			options=student_options,
			format_func=lambda x: f"{x[0]} - {x[1]}" if x[0] else "",
		)
		new_photo_file = st.file_uploader(
			"Upload New Photo",
			type=["jpg", "jpeg", "png"],
			key="manage_photo_id_uploader",
		)

		if st.button("Update Photo", key="manage_photo_update_btn"):
			if not selected_student or not selected_student[0]:
				st.error("Please select a valid student.")
			elif new_photo_file is None:
				st.error("Please upload a photo.")
			else:
				conn = None
				try:
					photo_bytes = new_photo_file.getvalue()
					pil_img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
					rgb_img = np.array(pil_img)
					opencv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

					results = DeepFace.represent(
						img_path=opencv_img,
						model_name="Facenet",
						detector_backend="opencv",
						enforce_detection=True,
					)

					encoding = np.array(results[0]["embedding"], dtype=np.float64)
					encoding_blob = pickle.dumps(encoding)

					conn = get_connection()
					cursor = conn.cursor()
					cursor.execute(
						"UPDATE students SET face_encoding = ?, photo_data = ? WHERE student_id = ?",
						(encoding_blob, photo_bytes, selected_student[0]),
					)
					conn.commit()

					st.success(f"Photo ID updated for {selected_student[0]}.")
					st.rerun()
				except ValueError:
					st.error("No face detected in uploaded photo. Please upload a clear face image.")
				except Exception as exc:
					st.error(f"Failed to update photo: {exc}")
				finally:
					if conn is not None:
						conn.close()

