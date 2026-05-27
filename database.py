import sqlite3


DB_NAME = "nero_attendance.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                student_id TEXT UNIQUE,
                name TEXT,
                face_encoding BLOB,
                photo_data BLOB
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exam_sessions (
                id INTEGER PRIMARY KEY,
                exam_code TEXT UNIQUE,
                subject TEXT,
                exam_date TEXT,
                exam_time TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance_roster (
                id INTEGER PRIMARY KEY,
                student_id TEXT,
                exam_code TEXT,
                table_number INTEGER,
                status TEXT DEFAULT 'Absent',
                timestamp DATETIME
            )
            """
        )

        # Backfill schema for existing databases created before photo_data existed.
        col_rows = cursor.execute("PRAGMA table_info(students)").fetchall()
        existing_cols = {row[1] for row in col_rows}
        if "photo_data" not in existing_cols:
            cursor.execute("ALTER TABLE students ADD COLUMN photo_data BLOB")

        conn.commit()
    finally:
        if conn is not None:
            conn.close()
