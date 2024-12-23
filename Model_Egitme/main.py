import cv2
from ultralytics import YOLO
import os

# Model ve görüntü yolu
model_path = "best.pt"
image_path = "buoy.jpg"

# Dosyaların varlığını kontrol et
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

# Modeli yükle
model = YOLO(model_path)

# Tahmin yap
results = model.predict(source=image_path, save=True, conf=0.25)

# Görüntüyü yükle
image = cv2.imread(image_path)

# Tespit edilen nesneler üzerinde işlem yap
for result in results:
    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        x1, y1, x2, y2 = map(int, box)  # Çerçeve koordinatlarını tam sayı yap
        confidence = float(conf)  # Güven skoru
        label = result.names[int(cls)]  # Nesne sınıfı adı

        # Çerçeve çiz
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil çerçeve
        cv2.putText(
            image,
            f"{label} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )  # Çerçevenin üzerine metin yaz

# İşlenmiş görüntüyü kaydet
output_path = "output_image.jpg"
cv2.imwrite(output_path, image)
print(f"Sonuç görüntüsü kaydedildi: {output_path}")

# İşlenmiş görüntüyü göster
cv2.imshow("Sonuç Görüntüsü", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
