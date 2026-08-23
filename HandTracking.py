import cv2
import mediapipe as mp

# MediaPipe el takibi modüllerini başlat
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Kamerayı başlat (0 genelde dahili web kamerasındır, harici kullanıyorsan 1 veya 2 yapabilirsin)
cap = cv2.VideoCapture(0)

print("Kamera açılıyor... Çıkmak için kameradayken 'q' tuşuna bas.")

# MediaPipe modelini yapılandır
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            continue

        # Görüntüyü aynala (sağ-sol karışıklığını önlemek için) ve RGB'ye çevir
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Görüntüyü işle ve elleri bul
        results = hands.process(image_rgb)

        # Eğer ekranda bir el tespit edildiyse
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # İleride motoru yönlendirmek için kullanacağımız hedef nokta: İşaret Parmağı Ucu (Landmark 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Koordinatları 0-1 aralığından ekran piksellerine çevir
                h, w, c = image.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Tüm iskeleti çiz
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Hedef noktamızı (İşaret parmağı ucu) devasa mavi bir daire ile vurgula
                cv2.circle(image, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                
                # Ekrana X ve Y değerlerini yazdır (Motor için asıl ihtiyacımız olan veri bu)
                cv2.putText(image, f"Hedef X: {cx} Y: {cy}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Sonuç görüntüsünü ekranda göster
        cv2.imshow('El Takip Sistemi', image)

        # 'q' tuşuna basılırsa döngüyü kır
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()