import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    # Попытка загрузки изображения
    face = cv2.imread("image/1.jpg")

    if face is None:
        raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")

    print(type(face))
    print(face.shape)

    # Конвертация изображения из BGR в RGB
    rgb_img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Отображение изображения с использованием Matplotlib
    plt.imshow(rgb_img)
    plt.axis('off')  # Скрыть оси
    plt.show()

except FileNotFoundError as e:
    print(e)
except cv2.error as e:
    print(f"Ошибка при работе с OpenCV: {e}")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")
