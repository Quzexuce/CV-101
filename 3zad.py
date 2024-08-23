import cv2

try:
    # Загрузить предобученный классификатор для детекции лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Открыть веб-камеру
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Не удалось открыть веб-камеру")

    while True:
        # Захватить кадр
        ret, frame = cap.read()
        if not ret:
            print("Не удалось захватить кадр")
            break

        # Преобразовать кадр в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружить лица
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Нарисовать прямоугольники вокруг обнаруженных лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Отобразить кадр с прямоугольниками
        cv2.imshow('Webcam - Face Detection', frame)

        # Ожидание нажатия клавиши 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Произошла ошибка: {e}")

finally:
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
