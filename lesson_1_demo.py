import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("Images\\Outdoor-Toy-Storage.jpg")

gray_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.imshow(gray_frame, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("Backyard")

plt.show()