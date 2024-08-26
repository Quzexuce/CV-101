import cv2
import face_recognition
import numpy as np
import time

# Загрузка известных изображений лиц и их кодировка
known_face_encodings = []
known_face_names = []

# Пример загрузки и кодировки лица
image1 = face_recognition.load_image_file("image/2.jpg")
face_encoding1 = face_recognition.face_encodings(image1)[0]
known_face_encodings.append(face_encoding1)
known_face_names.append("Dmitry")

image2 = face_recognition.load_image_file("image/3.jpg")
face_encoding2 = face_recognition.face_encodings(image2)[0]
known_face_encodings.append(face_encoding2)
known_face_names.append("Ilya")

# Инициализация захвата видео с веб-камеры
video_capture = cv2.VideoCapture(0)

while True:
    # Захват кадра видео
    ret, frame = video_capture.read()

    if not ret:
        print("Не удалось захватить кадр с камеры")
        break

    # Проверка типа данных и размерности кадра
    if frame.dtype != np.uint8:
        print("Ошибка: некорректный тип данных изображения")
        break

    if frame.ndim != 3 or frame.shape[2] != 3:
        print("Ошибка: некорректная размерность изображения")
        break

    # Преобразование изображения из BGR (по умолчанию в OpenCV) в RGB
    rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
    # Обнаружение лиц на текущем кадре видеоq
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Замер времени обнаружения лиц
    start_time_detection = time.time()
    face_locations = face_recognition.face_locations(rgb_frame)
    detection_time = time.time() - start_time_detection

    # Замер времени кодирования лиц
    start_time_encoding = time.time()
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    encoding_time = time.time() - start_time_encoding

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Используем известное лицо с наименьшим расстоянием до нового лица
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Отрисовка рамки вокруг лица
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Отрисовка метки с именем ниже лица
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Отображение замеров времени
    cv2.putText(frame, f'Detection Time: {detection_time:.3f} sec', (10, 30), font, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f'Encoding Time: {encoding_time:.3f} sec', (10, 60), font, 0.6, (0, 255, 0), 1)

    cv2.imshow('Video', frame)

    # Завершение работы при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()
