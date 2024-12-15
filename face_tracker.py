import cv2
import numpy as np
import os

def main():
    print(os.path.exists('C://Users//SHAWRYA//PycharmProjects//ComputerVision//data//haarcascade_frontalface_default.xml'))

    #load cascades
    face_cascade=cv2.CascadeClassifier('C://Users//SHAWRYA//PycharmProjects//ComputerVision//data//haarcascade_frontalface_default.xml')
    eye_cascade=cv2.CascadeClassifier('C://Users//SHAWRYA//PycharmProjects//ComputerVision//data//haarcascade_eye.xml')

    if face_cascade.empty():
        print("Error loading face cascade")
        exit()

    if eye_cascade.empty():
        print("Error loading eye cascade")
        exit()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    # Start video capture
    print("Press 'w' to stop the program.")
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            print(f"Faces detected: {len(faces)}")

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Detect eyes within each face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
                print(f"Eyes detected: {len(eyes)} within face region")

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(roi_color, 'Eye', (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display the video frame
            cv2.imshow('Face & Eye Detection', img)

            # Exit loop on 'w' key press
            if cv2.waitKey(1) & 0xFF == ord('w'):
                print("Program stopped by user.")
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released. Program exited.")
if __name__ == "__main__":
    main()