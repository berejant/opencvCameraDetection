#!/usr/bin/env python3
import os
import cv2
import urllib.request

repo = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/data/haarcascades/"

for file in ["haarcascade_frontalface_default.xml", "haarcascade_eye.xml"]:
  if not os.path.exists(file):
    urllib.request.urlretrieve(repo + file, file)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Initialize the camera
cam = cv2.VideoCapture(0)

if cam is None:
    print("Camera not found")
    exit(1)

# Start an infinite loop for real-time image processing
while True:
    # Read a frame from the camera
    status, photo = cam.read()
    if not status:
        print("Failed to read camera")
        continue

    # convert to gray scale of each frames
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    print("Found %d faces" % len(faces))

    if len(faces) > 0:
        x, y, w, h = faces[0]

        cv2.rectangle(photo, (x, y), (x + w, y + h), (255, 255, 0), 2)

        face_gray = gray[y:y+h, x:x+w]
        face_color = photo[y:y+h, x:x+w]

        photo[0:h, 0:w] = face_color

        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 5)
        #To draw a rectangle in eyes
        for (ex,ey,ew,eh) in eyes:
          cv2.rectangle(photo,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

    # Display the modified image
    cv2.imshow("myphoto", photo)

    # Check for the 'Enter' key press to exit the loop
    if cv2.waitKey(10) == 13:
        break

# Release the camera and close all windows
cv2.destroyAllWindows()
cam.release()
