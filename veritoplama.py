import cv2
import os

# Kamera başlat
cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Sayısal ID girilmesini iste
while True:
    face_id = input("\nSayısal bir kullanıcı ID girin (örn: 1): ")
    if face_id.isdigit():
        face_id = int(face_id)
        break
    else:
        print("Lütfen sadece sayı girin.")

print("\nYüz toplama başlıyor. Çıkmak için 'q' tuşuna basın.")
count = 0
os.makedirs("dataset", exist_ok=True)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Veri Toplama', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

cam.release()
cv2.destroyAllWindows()
