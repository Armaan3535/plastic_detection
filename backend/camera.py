import cv2
import requests
import time

API_URL = "http://127.0.0.1:8000/predict"

cap = cv2.VideoCapture(1)

print("Camera starting...")
time.sleep(5) 

ret, frame = cap.read()

if ret:
    cv2.imwrite("temp.jpg", frame)

    with open("temp.jpg", "rb") as f:
        response = requests.post(API_URL, files={"file": f})

    result = response.json()
    print("Prediction Result:", result)

    cv2.imshow("Captured Image", frame)
    cv2.waitKey(3000) 

else:
    print("Failed to capture image")

cap.release()
cv2.destroyAllWindows()
