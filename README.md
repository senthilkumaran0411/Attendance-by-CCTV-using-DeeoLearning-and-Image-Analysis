# Face Recognition Based Attendance System

This project is a face recognition-based attendance system that uses OpenCV and Python. The system uses a camera to capture images of individuals and then compares them with the images in the database to mark attendance.

## Installation

1. Clone the repository to your local machine. ``` git clone https://github.com/Arijit1080/Face-Recognition-Based-Attendance-System ```
2. Install the required packages using ```pip install -r requirements.txt```.
3. Download the dlib models from https://drive.google.com/drive/folders/12It2jeNQOxwStBxtagL1vvIJokoz-DL4?usp=sharing and place the data folder inside the repo

## Usage

1. Collect the Faces Dataset by running ``` python get_faces_from_camera_tkinter.py``` .
2. Convert the dataset into ```python features_extraction_to_csv.py```.
3. To take the attendance run ```python attendance_taker.py``` .
4. Check the Database by ```python app.py```.


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have any suggestions.


Additional Feature for Bonus points: Proceed to ML. ops, and store the data into a DB So it can further processed and can reach a lot of devices

## Conditions
 the video quality you train the model should match with cetv. And No Machine Iguage should be used. Only Deep Learning should be used

## Tips
 Use opencv Face cascade to detect faces. You should train the model, for better quality output.

TestCases: We will provide a video of 5 people walking inside the classroom. Grades depend on successful outcomes.


