# Face Recognition Attendance System

A real-time Computer Vision application built with Python that utilizes automated facial recognition to track and log attendance into a CSV database.

## 🚀 How It Works
1. **Face Data Ingestion:** The system scans the `images/` directory to load reference photos and automatically extracts employee/student names from the file metadata.
2. **Mathematical Encoding:** It processes face locations and converts facial landmarks into distinct biometric encodings using a pre-trained neural network framework.
3. **Live Camera Processing:** BGR video frames from the webcam are downscaled to 25% for performance optimization, converted to RGB, and matched against known profiles using Euclidean distance optimization (`numpy.argmin`).
4. **Automated Logging:** Verified matches trigger a user interface overlay (bounding box + name tag) and register a real-time timestamped entry into `attendance.csv` if they are not already marked present.

## 🛠️ Tech Stack & Tools
* **Python 3.11** - Main programming environment
* **OpenCV (`cv2`)** - Live webcam capture and video frame manipulation
* **Face_Recognition** - High-level wrapper built on top of dlib for facial land-marking and tracking
* **NumPy & Pandas** - Array calculations and database formatting
* **Dlib-bin** - Pre-compiled binary engine for Windows compatibility

## 📋 File Architecture
```text
Attendance_Project/
│
├── images/                  # Reference photos named as 'person_name.jpg'
│   └── marvex.jpg           
│
├── main.py                  # Core application script with custom logic
├── attendance.csv           # Live log spreadsheet tracking Name and Timestamp
└── README.md                # Project documentation
```

## ⚙️ Quick Installation & Setup

1. Clone or download this project folder to your machine.
2. Set up your Python environment and run the package dependencies:
   ```bash
   pip install opencv-python dlib-bin face-recognition numpy pandas Pillow Click
   ```
3. Drop a clear front-facing headshot into the `images/` folder and name the file `your_name.jpg`.
4. Fire up the application:
   ```bash
   python main.py
   ```
5. Press the **'q'** key on your keyboard to turn off the video feed and save your logs safely.
