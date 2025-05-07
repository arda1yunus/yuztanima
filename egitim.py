import cv2
import numpy as np
import os

# LBPH tanıyıcıyı oluştur
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for image_path in image_paths:
        try:
            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                print(f"Hatalı dosya atlandı (okunamadı): {image_path}")
                continue

            faces = detector.detectMultiScale(gray_img)
            if len(faces) == 0:
                print(f"Hatalı dosya atlandı (yüz bulunamadı): {image_path}")
                continue

            filename = os.path.split(image_path)[-1]
            id_str = filename.split(".")[1]
            id_num = int(id_str)

            for (x, y, w, h) in faces:
                face_samples.append(gray_img[y:y+h, x:x+w])
                ids.append(id_num)

        except Exception as e:
            print(f"Hatalı dosya atlandı: {image_path} ({str(e)})")

    return face_samples, ids

print("Model eğitiliyor...")
faces, ids = get_images_and_labels('dataset')

if len(faces) < 2:
    print("Yeterli yüz verisi yok. Eğitim başarısız.")
else:
    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')
    print("Model başarıyla 'trainer.yml' dosyasına kaydedildi.")
