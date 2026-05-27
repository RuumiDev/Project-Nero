# Project Nero

> **Face-Recognition Exam Attendance System**
> Made with love by **RuumiDev** ~!

---

## Table of Contents

- [Project Nero](#project-nero)
  - [Table of Contents](#table-of-contents)
  - [Objective](#objective)
  - [Problem Statement](#problem-statement)
  - [System Workflow](#system-workflow)
  - [Features](#features)
    - [Student Kiosk](#student-kiosk)
    - [Admin Dashboard](#admin-dashboard)
    - [Student Enrollment](#student-enrollment)
  - [Tech Stack \& Frameworks](#tech-stack--frameworks)
    - [Recognition Model Details](#recognition-model-details)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the App](#running-the-app)
    - [First-Time Setup](#first-time-setup)
  - [Production Considerations \& Technical Debt](#production-considerations--technical-debt)
  - [Future Upgrades](#future-upgrades)
    - [Security \& Auth](#security--auth)
    - [Recognition \& Liveness](#recognition--liveness)
    - [Data \& Infrastructure](#data--infrastructure)
    - [UI / UX](#ui--ux)
    - [Operations](#operations)
  - [Security Notes](#security-notes)

---

## Objective

Project Nero is a **biometric exam attendance system** designed for academic institutions. It replaces traditional paper-based or manual sign-in sheets with an automated, camera-driven kiosk that identifies students via real-time facial recognition and marks their attendance in a centralized database — all without requiring students to carry any physical ID or card.

The system is built for two audiences:

| Role | Interface | Purpose |
|---|---|---|
| **Student** | Live Kiosk (`1_Student_Kiosk.py`) | Walk up, look at the camera, get verified and seated |
| **Administrator** | Dashboard (`2_Admin_Dashboard.py`) | Manage students, schedule exams, track attendance, export reports |

---

## Problem Statement

Exam attendance in many academic environments still relies on manual processes:

- **Sign-in sheets** that can be signed by proxy or forged
- **Manual ID checks** that are slow, error-prone, and require staff at every entry point
- **No real-time visibility** into who is present until sheets are collected and tallied after the exam

These processes create opportunities for impersonation, delays during check-in, and administrative overhead in report generation.

**Project Nero** addresses this by:

- Removing the need for any physical card or QR code — identity is the face
- Providing **real-time liveness detection** (blink-to-verify) to prevent photo spoofing
- Giving administrators an instant live view of attendance status per exam session
- Automatically generating exportable attendance rosters

---

## System Workflow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          ENROLLMENT (Admin Side)                             │
│                                                                              │
│  Admin Dashboard → Register Student → Capture Face Photo →                  │
│  DeepFace (FaceNet) generates 128-d embedding → Stored in SQLite DB         │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          EXAM SESSION SETUP (Admin)                          │
│                                                                              │
│  Create Exam Code → Set Subject / Date / Time →                             │
│  Assign students to exam + table numbers via the Master Grid                │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          EXAM DAY — STUDENT KIOSK                            │
│                                                                              │
│  Student approaches camera →                                                 │
│  1. MediaPipe liveness check (eye blink / EAR detection)                    │
│  2. DeepFace (FaceNet) cosine-distance match against enrolled encodings     │
│  3. Match found → roster record updated to "Present" + timestamp            │
│  4. Student profile + assigned table number displayed on screen             │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          REPORTING (Admin)                                   │
│                                                                              │
│  Admin Dashboard → Filter by Exam Code / Date →                             │
│  View Present / Absent counts → Export CSV Roster                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Student Kiosk
- **Live camera feed** with multi-backend fallback (DirectShow → MSMF → Default), scanning indices 0–3
- **Liveness detection** via Eye Aspect Ratio (EAR) using MediaPipe Face Mesh — student must blink before recognition fires
- Fallback to **MediaPipe Tasks API** when the legacy `mp.solutions` API is unavailable
- **Frame skipping** (every N frames) to reduce CPU/GPU load during recognition
- On successful match: displays student name, photo, exam code, and assigned table number for a configured hold period
- Graceful degradation — kiosk runs in recognition-only mode if MediaPipe is unavailable

### Admin Dashboard
- **Password-protected** sidebar gate (configurable)
- **Master Grid** — unified view of all students, exam sessions, and attendance status via a live SQLite join
- **Inline editing** — edit student names, exam codes, exam dates, and table numbers directly in the grid; save to DB in one click
- **Add / delete rows** inline — automatically inserts or removes records from `students`, `attendance_roster`, and `exam_sessions`
- **Multi-filter** — filter the grid by exam code and/or exam date
- **Live metrics** — total students and present count per filtered view
- **CSV export** — one-click download of the official attendance roster

### Student Enrollment
- Capture student photo via webcam or upload
- Face embedding generated by DeepFace (FaceNet model)
- Embedding stored as a binary blob alongside student ID and photo

---

## Tech Stack & Frameworks

| Layer | Technology | Purpose |
|---|---|---|
| **UI / App server** | [Streamlit](https://streamlit.io/) | Multi-page web UI, real-time video display, data editor |
| **Face Recognition** | [DeepFace](https://github.com/serengil/deepface) (FaceNet model) | 128-d face embedding generation & cosine-distance matching |
| **Liveness Detection** | [MediaPipe](https://developers.google.com/mediapipe) | Face mesh landmark extraction, Eye Aspect Ratio (EAR) blink detection |
| **Computer Vision** | [OpenCV](https://opencv.org/) | Camera capture, frame processing, bounding box rendering |
| **Database** | SQLite (via Python `sqlite3`) | Lightweight, file-based, zero-config persistence |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) | DataFrame manipulation, embedding math |
| **Image Handling** | [Pillow (PIL)](https://pillow.readthedocs.io/) | Photo decoding and display |
| **Serialization** | Python `pickle` | Binary face encoding storage/retrieval |

### Recognition Model Details

- **Model**: FaceNet (via DeepFace)
- **Detector backend**: OpenCV (fast, CPU-friendly)
- **Distance metric**: Cosine distance
- **Match threshold**: `0.4` — embeddings within this distance are considered the same person
- **Frame processing**: Every `5` frames to balance accuracy and CPU load

---

## Project Structure

```
Project-Nero/
│
├── app.py                  # Legacy camera preview & recognition module
├── database.py             # DB connection factory + schema init (nero_attendance.db)
├── attendance_db.py        # Standalone DB initializer (nero_attendance.db)
│
├── pages/
│   ├── 1_Student_Kiosk.py  # Live kiosk: liveness + face recognition + mark present
│   └── 2_Admin_Dashboard.py# Admin: manage students, exams, roster, export CSV
│
├── .gitignore              # Excludes DB files, venv, model cache, secrets
└── README.md               # This file
```

> **Not committed to git** (see `.gitignore`):
> - `nero.db` / `nero_attendance.db` — contain student biometric data
> - `.venv/` — Python virtual environment
> - `.model_cache/` — downloaded MediaPipe `.task` model files
> - `__pycache__/`

---

## Getting Started

### Prerequisites

- Python 3.9+
- A webcam (index 0 by default; falls back to indices 1–3)
- Windows (DirectShow backend used first), Linux/macOS also supported

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/RuumiDev/Project-Nero.git
cd Project-Nero

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install streamlit deepface mediapipe opencv-python-headless pandas numpy pillow
# Note: use opencv-python instead of opencv-python-headless if you need GUI windows
```

### Running the App

```bash
streamlit run app.py
```

Streamlit will open the app in your browser. Navigate using the sidebar:
- **Student Kiosk** — activate the kiosk camera to begin recognition
- **Admin Dashboard** — enter the admin password to manage data

### First-Time Setup

1. Go to **Admin Dashboard**
2. Enroll students: enter Student ID, name, and capture a face photo
3. Create an exam session: set exam code, subject, date, and time
4. Assign students to the exam roster with table numbers
5. On exam day, activate the **Student Kiosk** — students approach the camera and blink to check in

---

## Production Considerations & Technical Debt

| Constraint | Description |
|---|---|
| **Hardcoded admin password** | The admin password (`admin123`) is currently set as a literal string in `2_Admin_Dashboard.py`. It is not hashed or stored securely. |
| **Single-camera kiosk** | The kiosk uses `cv2.VideoCapture(0)` as a fixed index for the exam-day stream; the enrollment flow uses a fallback loop across indices 0–3. |
| **No multi-user auth** | There is no user account system — a single shared admin password gates the entire dashboard. |
| **SQLite concurrency** | SQLite has limited write concurrency. If multiple admins access the dashboard simultaneously, write conflicts may occur. |
| **Local file storage** | Databases and model files are stored on disk locally. There is no cloud sync, backup, or redundancy. |
| **CPU-bound recognition** | DeepFace inference runs on CPU by default. On low-spec hardware, frame latency may increase. |
| **Liveness via blink only** | The current anti-spoofing mechanism is an EAR blink check. A printed photo of someone blinking cannot pass, but a video replay might. |
| **No duplicate check-in guard** | If a student's record is already "Present", the system will still fire the recognition pipeline (though the DB update is idempotent). |
| **No session isolation** | The app does not distinguish between multiple simultaneous exam sessions happening in different rooms on the same day. |
| **Model download on first run** | The MediaPipe `face_landmarker.task` model (~30 MB) is downloaded at runtime if not cached. Requires internet on first use. |

---

## Future Upgrades

### Security & Auth
- [ ] Replace hardcoded password with hashed credentials stored in a secure config file (e.g., bcrypt + `.env`)
- [ ] Add role-based access control (Super Admin vs. Invigilator)
- [ ] Implement session tokens / JWT for multi-user dashboard access
- [ ] Add HTTPS/TLS for any networked deployment

### Recognition & Liveness
- [ ] GPU acceleration for DeepFace inference (CUDA/DirectML)
- [ ] Upgrade liveness to a multi-frame challenge-response (head turn, smile) or integrate a dedicated anti-spoofing model
- [ ] Support multiple simultaneous face detections per frame (crowd-mode for entry queues)
- [ ] Configurable match threshold per deployment environment

### Data & Infrastructure
- [ ] Migrate from SQLite to PostgreSQL for multi-user concurrent access
- [ ] Cloud storage for face encoding blobs (AWS S3 / Azure Blob)
- [ ] Automated database backups
- [ ] Audit log — record every attendance action with who triggered it

### UI / UX
- [ ] Student self-enrollment portal with email verification
- [ ] Real-time dashboard refresh (WebSocket or Streamlit auto-rerun)
- [ ] Room / hall management — multiple exam rooms per session
- [ ] Printable seating charts generated from roster data

### Operations
- [ ] Docker container with bundled dependencies for zero-config deployment
- [ ] CI/CD pipeline (GitHub Actions) for lint + test on push
- [ ] Unit tests for face matching logic, DB operations, and EAR calculation
- [ ] Configuration file (`config.toml` or `.env`) for threshold, camera index, admin credentials

---

## Security Notes

The following files are excluded from version control via `.gitignore` because they contain sensitive information:

| File / Directory | Reason |
|---|---|
| `nero.db` | Contains student face embeddings (biometric PII) |
| `nero_attendance.db` | Contains attendance records and face embeddings |
| `.env` / `secrets.py` | Intended for future credential storage |
| `.model_cache/` | Large binary model files (not sensitive, but large) |
| `.venv/` | Local Python environment |

> **Note on the admin password**: The current password (`admin123`) in `2_Admin_Dashboard.py` is a placeholder. **Change it before any real deployment** and migrate to a hashed credential system.

---

*Made with love by **RuumiDev** ~!*
