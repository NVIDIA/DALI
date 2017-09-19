import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

img = []
width = 0;
height = 0;
with open(argv[1], 'r') as file:
    for line in file:
        a = line.split()

        # pull width
        if width == 0:
            width = len(a)

        a = list(map(float, a))
        img.append(a)
        height += 1

img = np.asarray(img)
print(img.shape)
img = img.reshape([height, width])
img = img.astype(np.uint8)

plt.imshow(img, cmap='gray')
plt.show()
