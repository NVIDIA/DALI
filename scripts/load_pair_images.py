import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from load_image import load_image

res, res_cmap = load_image(argv[1])
ver, ver_cmap = load_image(argv[2])
assert res.shape == ver.shape

# Calculate the absolute difference
diff = np.absolute(np.subtract(res.astype(np.float), ver.astype(np.float)))
print("max: " + str(np.max(diff)))
print("min: " + str(np.min(diff)))
print("mean: " + str(np.mean(diff)))
print("var: " + str(np.var(diff)))

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(res, cmap = res_cmap)
ax2.imshow(ver, cmap = ver_cmap)
ax3.imshow(diff, cmap = 'gray')
plt.show()
