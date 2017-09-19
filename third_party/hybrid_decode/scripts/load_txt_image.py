import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

img = []
with open(argv[1], 'r') as file:
    for line in file:
        a = line.split()
        a = list(map(float, a))
        img.append(a)

img = np.asarray(img)
print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()
