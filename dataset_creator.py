import cv2
import os

# Create dataset folder if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

face_id = input("Enter User ID: ")

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

count = 0

print("Capturing images... Look at the camera")

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        count += 1
        print("Saving image:" ,count)
                    
        cv2.imwrite(f"dataset/user.{face_id}.{count}.jpg", gray[y:y+h,x:x+h])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('image', img)

    if count >= 50:
        break

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cam.release()
cv2.destroyAllWindows()

print("Dataset Created Successfully!")