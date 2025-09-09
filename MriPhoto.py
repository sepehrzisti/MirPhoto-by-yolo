import cv2
import numpy as np
import matplotlib.pyplot as plt

# گرفتن ویدئو از وبکم لپ‌تاپ (معمولاً index=0)
cap = cv2.VideoCapture(0)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # خاکستری کردن تصویر
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ساخت مختصات
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray

    # نمایش سه‌بعدی
    ax.clear()
    ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.draw()
    plt.pause(0.001)

cap.release()
cv2.destroyAllWindows()
