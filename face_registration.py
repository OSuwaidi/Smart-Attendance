# face_registration.py

import pickle
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

def save_face_data(name, faces_to_save):
    """
    Saves the captured face data and corresponding names to pickle files.
    """
    try:
        if not name or not faces_to_save:
            return False, "Name or face data is missing."

        faces_data = np.asarray(faces_to_save)
        faces_data = faces_data.reshape(len(faces_to_save), -1)

        if not os.path.exists('Data'):
            os.makedirs('Data')

        # Handle names file
        try:
            with open('Data/names.pkl', 'rb') as f:
                names = pickle.load(f)
        except (FileNotFoundError, EOFError):
            names = []
        
        names.extend([name] * len(faces_to_save))
        with open('Data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

        # Handle faces file
        try:
            with open('Data/faces_data.pkl', 'rb') as f:
                existing_faces = pickle.load(f)
            all_faces = np.append(existing_faces, faces_data, axis=0)
        except (FileNotFoundError, EOFError):
            all_faces = faces_data
            
        with open('Data/faces_data.pkl', 'wb') as f:
            pickle.dump(all_faces, f)
            
        return True, "Data saved successfully."
    except Exception as e:
        return False, f"An unexpected error occurred while saving data: {e}"