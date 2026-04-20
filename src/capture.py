from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture(0)  # 0 = webcam principal

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara")
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Proyecto Kaizen - Test YOLOv8", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()