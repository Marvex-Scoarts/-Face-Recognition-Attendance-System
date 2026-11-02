import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# =====================================================================
# STEP 1: IMPORTING EMPLOYEE IMAGES & EXTRACTING NAMES
# =====================================================================
# Here we point the script to the folder containing reference photos
path = 'images'
known_images = []
known_names = []
my_list = os.listdir(path)

for cl in my_list:
    current_img = cv2.imread(f'{path}/{cl}')
    known_images.append(current_img)
    # This splits 'your_name.jpg' into just 'your_name'
    known_names.append(os.path.splitext(cl)[0])

# =====================================================================
# STEP 2: GENERATING FACE MAPS (ENCODINGS)
# =====================================================================
# This function translates the facial characteristics into unique mathematical profiles.
def find_encodings(images):
    encode_list = []
    for img in images:
        # This converts the image from BGR color (OpenCV default) to RGB color (Face_Recognition default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img_rgb)[0]
        encode_list.append(encode)
    return encode_list

print("AI is learning the faces... Please wait.")
known_encode_list = find_encodings(known_images)
print("Learning complete!")

# =====================================================================
# STEP 3: LOGGING SYSTEM FOR ATTENDANCE
# =====================================================================
# This function handles the opening of the log file and registering the timestamp
def mark_attendance(name):
    with open('attendance.csv', 'r+') as f:
        data_list = f.readlines()
        name_list = []
         # This will check who has already been logged today so we don't duplicate entries
        for line in data_list:
            entry = line.split(',')
            name_list.append(entry[0])
            
         # If the person is missing from the list, append their name and current time
        if name not in name_list:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time_string}')

# =====================================================================
# STEP 4: REAL-TIME WEBCAM PROCESSING LOOP
# =====================================================================
# Activate the primary default webcam
# '0' tells the computer to turn on your built-in laptop camera.

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # Shrink the camera image a bit so the AI can process it super fast
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    # Find any faces currently visible on the webcam
    faces_in_frame = face_recognition.face_locations(img_small_rgb)
    encodes_in_frame = face_recognition.face_encodings(img_small_rgb, faces_in_frame)

    # Compare the live webcam faces against the photos the AI learned earlier
    for encode_face, face_loc in zip(encodes_in_frame, faces_in_frame):
        matches = face_recognition.compare_faces(known_encode_list, encode_face)
        face_distance = face_recognition.face_distance(known_encode_list, encode_face)
        
        # Find the lowest distance (the best mathematical match)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = known_names[match_index].upper()
            
            # Draw a box around the face on screen and display the name
            y1, x2, y2, x1 = face_loc
            # Multiply by 4 to scale the box back up to full video size
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            # Call our function to write the name in the Excel file we created            
            mark_attendance(name)

    # Show the webcam window to the user
    cv2.imshow('Webcam Attendance System', img)
    
    # Press the 'q' key on your keyboard to turn off the camera and stop the app
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Safely shut down camera resources and dismiss standard system windows
cap.release()
cv2.destroyAllWindows()