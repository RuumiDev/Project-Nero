import sqlite3


def init_db():
    """Initialize the nero_attendance.db database and required tables."""
    conn = None
    try:
        conn = sqlite3.connect("nero_attendance.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                student_id TEXT UNIQUE,
                name TEXT,
                face_encoding BLOB
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

        conn.commit()
        print("Database initialized successfully: nero_attendance.db")

    except sqlite3.Error as exc:
        print(f"SQLite error while initializing database: {exc}")

    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    init_db()
