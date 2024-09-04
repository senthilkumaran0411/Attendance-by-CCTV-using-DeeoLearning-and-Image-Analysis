import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime

# Define paths for model files
shape_predictor_path = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
face_recognition_model_path = 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'

# Verify that model files exist
if not os.path.isfile(shape_predictor_path):
    raise RuntimeError(f"Unable to open {shape_predictor_path}. Please ensure the file exists.")

if not os.path.isfile(face_recognition_model_path):
    raise RuntimeError(f"Unable to open {face_recognition_model_path}. Please ensure the file exists.")

# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor(shape_predictor_path)

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1(face_recognition_model_path)

# Create a connection to the database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create a table for the current date
current_date = datetime.datetime.now().strftime("%Y_%m_%d")  # Replace hyphens with underscores
table_name = "attendance"
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
cursor.execute(create_table_sql)

# Commit changes and close the connection
conn.commit()
conn.close()


class Face_Recognizer:
    # ... (rest of your class remains unchanged)

    def __init__(self):
        # Initialization code remains unchanged
        pass

    # ... (other methods remain unchanged)

    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)  # Get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
