# attendance_system.py

import pickle
import os
import csv
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

def load_model():
    """
    Loads face data and trains a KNN classifier.
    Returns a tuple: (classifier, error_message).
    """
    try:
        with open('Data/names.pkl', 'rb') as w:
            LABELS = pickle.load(w)
        with open('Data/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(FACES, LABELS)
        # ** FIX: Return the model and None for the error message **
        return knn, None
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        # Return None for the model and the error message string
        return None, f"Error loading model data: {e}. Please register a face first."

def mark_attendance(name):
    """
    Marks attendance for a given name by appending it to today's CSV file.
    """
    try:
        print('Entered the function')
        ts = datetime.now(ZoneInfo("Asia/Kolkata"))
        date = ts.strftime("%d-%m-%Y")
        timestamp = ts.strftime("%H:%M:%S")

        # Base directory of this script
        base_dir = Path(__file__).parent.resolve()
        attendance_dir = base_dir / "Attendance"

        # Ensure Attendance directory exists
        attendance_dir.mkdir(exist_ok=True)

        filename = attendance_dir / f"Attendance_{date}.csv"

        attendance_record = [name, timestamp]

        file_exists = filename.is_file()

        if file_exists:
            with open(filename) as f:
                lines = f.read().splitlines()[1:]  # skip header
            if any(name == line.split(',')[0] for line in lines):
                print('Already marked')
                return "This person's attendance has already been taken"

        with filename.open("a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['NAME', 'TIME'])  # Write header if new file
            writer.writerow(attendance_record)
        print('Noted')
        return f"Attendance marked for {name} at {timestamp}"

    except Exception as e:
        return f"Error marking attendance: {e}"