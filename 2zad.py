import cv2

try:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Не удалось открыть веб-камеру")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось захватить кадр")
            break

        

        # Ожидание нажатия клавиши 's' для выхода
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

except Exception as e:
    print(f"Произошла ошибка: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
