import cv2
import numpy as np
from sys import argv
import matplotlib.pyplot as plt

ver = argv[1]
ver = cv2.imread(ver)
ver = cv2.cvtColor(ver, cv2.COLOR_BGR2RGB)

# plt.imshow(ver)
# plt.show()

img = []
width = 0;
height = 0;
components = 0;
n = 0
with open(argv[2], 'r') as file:
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
img = img.reshape([height, width, components])
img = img.astype(np.uint8)


print(img.shape)
# print(img.dtype)
# print(ver.shape)
# print(ver.dtype)

diff = np.absolute(np.subtract(img.astype(np.float), ver.astype(np.float)))
# diff = np.subtract(img.astype(np.float), ver.astype(np.float))
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

print(diff[0, :, 0])
print(img[0, :, 0])
print(ver[0, :, 0])

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(ver)
ax2.imshow(img)
plt.show()
