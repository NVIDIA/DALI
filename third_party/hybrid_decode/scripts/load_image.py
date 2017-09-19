import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

img = []
width = 0;
height = 0;
components = 0;
n = 0
with open(argv[1], 'r') as file:
    for line in file:
        a = line.split()
        if n == 0:
            assert(len(a) == 3)
            height = int(a[0])
            width = int(a[1])
            components = int(a[2])
            n = 1
            continue
            
        a = list(map(float, a))
        img.append(a)

img = np.asarray(img)
if components == 3:
    img = img.reshape([height, width, components])
img = img.astype(np.uint8)
print (img.shape)

if components == 1:
    plt.imshow(img, cmap='gray')
else:
    plt.imshow(img)
# plt.xlim(0, width)
# plt.ylim(0, height)
plt.show()



