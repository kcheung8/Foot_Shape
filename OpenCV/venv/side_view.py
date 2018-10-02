import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


img = cv2.imread('A4Paper17.jpg',0)
gray = cv2.GaussianBlur(img, (11, 11), 0)

#edges = cv2.Canny(gray,100,200)
edge_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)


# Eliminate zero values with method 2
pos_edge_x = np.abs(edge_x) / np.max(np.abs(edge_x))
cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
cv2.waitKey(0)


plt.gray()
plt.subplot(1,2,1)
plt.imshow(pos_edge_x)

plt.subplot(1,2,2)
plt.imshow(gray)
plt.show()