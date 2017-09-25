import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from load_image import load_image

res, res_cmap = load_image(argv[1])
ver, ver_cmap = load_image(argv[2])
assert res.shape == ver.shape

# for i in xrange(res.shape[0]):
#     for j in xrange(res.shape[1]):
#         for k in xrange(res.shape[2]):
#             a = float(res[i, j, k])
#             b = float(ver[i, j, k])
#             print ("h,w,c: %d,%d,%d" % (i, j, k))
#             print ("res v. ver: %d,%d" % (a, b))
#             print("diff: %f" % (abs(a-b)))

# Calculate the absolute difference
diff = np.absolute(np.subtract(res.astype(np.float), ver.astype(np.float)))
print("max: " + str(np.max(diff)))
print("min: " + str(np.min(diff)))
print("mean: " + str(np.mean(diff)))
print("std: " + str(np.std(diff)))

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
ax1.imshow(res, cmap = res_cmap)
ax2.imshow(ver, cmap = ver_cmap)
ax4.imshow(diff[:,:,0], cmap = 'gray')
ax5.imshow(diff[:,:,1], cmap = 'gray')
ax6.imshow(diff[:,:,2], cmap = 'gray')
plt.show()
