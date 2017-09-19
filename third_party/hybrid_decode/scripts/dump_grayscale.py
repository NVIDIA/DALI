import cv2
import numpy as np
from sys import argv
import matplotlib.pyplot as plt

ver = argv[1]
ver = cv2.imread(ver)
ver = cv2.cvtColor(ver, cv2.COLOR_BGR2GRAY)

# plt.imshow(ver)
# plt.show()

img = []
with open(argv[2], 'r') as file:
    for line in file:
        a = line.split()
        a = list(map(float, a))
        img.append(a)

img = np.asarray(img)
img = img.astype(np.uint8)
img = img[:, 0:75]


print(img.shape)
# print(img.dtype)
# print(ver.shape)
# print(ver.dtype)

diff = np.absolute(np.subtract(img.astype(np.float), ver.astype(np.float)))
# print(diff[0, 0])
# print(img[0,0])
# print(ver[0,0])
# print(diff)
print("max: " + str(np.max(diff)))
print("min: " + str(np.min(diff)))
print("mean: " + str(np.mean(diff)))
print("var: " + str(np.var(diff)))
# a = np.nonzero(diff)
# print(a)

print(diff[0, :])
print(img[0, :])
print(ver[0, :])

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(ver, cmap='gray')
ax2.imshow(img, cmap='gray')
plt.show()
