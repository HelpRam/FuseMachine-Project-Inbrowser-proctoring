import cv2
import numpy as np
import datetime
from facial_detections import detectFace
# from eye_tracker import gazeDetection
# from blink_detection import isBlinking
from head_pose_estimation import head_pose_detection
from mouth_tracking import mouthTrack
from object_detection import detectObject
# from audio_detection import audio_detection

# Initialize video capture (0 for default camera, or provide a video file path)
cap = cv2.VideoCapture(0)

# Log file path
log_file = "activity_log.txt"

def log_activity(log_file, activity):
    with open(log_file, 'a') as file:
        file.write(str(activity) + '\n')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Capture the current time for logging
    current_time = str(datetime.datetime.now().time())

    # Detect faces
    faceCount, faces = detectFace(frame)
    face_status = "Face detecting properly." if faceCount > 0 else "No face detected."

    # blinkStatus = "Not checked"
    # gazeDirection = "Not checked"
    headPose = "Not checked"
    mouthStatus = "Not checked"

    if faceCount > 0:
        # Perform eye gaze detection
        # gazeDirection = gazeDetection(faces, frame)

        # Detect blinks
        # blinkStatus = isBlinking(faces, frame)

        # Perform head pose estimation
        headPose = head_pose_detection(faces, frame)

        # Perform mouth tracking
        mouthStatus = mouthTrack(faces, frame)

    # Perform object detection
    detectedObjects = detectObject(frame)

    # Perform audio detection
    # audioStatus = audio_detection()

    # Combine the statuses into one activity log entry
    activity = [current_time, face_status, mouthStatus, detectedObjects, headPose]

    # Log the activity
    log_activity(log_file, activity)

    # Display the frame with annotations
    cv2.imshow('Facial Detection', frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
