import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids = [], []

    for image_path in image_paths:
        gray_img = Image.open(image_path).convert('L')
        img_np = np.array(gray_img, 'uint8')
        id_ = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(id_)
    return face_samples, ids

print("\nModel eğitiliyor...")
faces, ids = get_images_and_labels('dataset')
recognizer.train(faces, np.array(ids))
recognizer.write('trainer.yml')
print(f"\n{len(np.unique(ids))} kişi için model eğitildi.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

names = ['Unknown', 'Kullanici1', 'Kullanici2']  # ID'lere karşılık gelen isimler

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        name = names[id_] if confidence < 70 else 'Bilinmiyor'
        confidence_text = f"%{round(100 - confidence)} doğruluk"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence_text), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Yüz Tanıma', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
