import cv2
import matplotlib.pyplot as plt

from thresholding import region_growing, get_segments
from utils import rgb2grey, read_image

img_samp_color = cv2.cvtColor(read_image('obrazy\o3.jpg'), cv2.COLOR_BGR2RGB)
img_samp = rgb2grey(img_samp_color)
#img_samp_color = median_filter(img_samp_color, filter_size=3)

segmat, markers = region_growing(img_samp_color, threshold=10, neighbourhood=4)

get_segments(segmat)

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 15))
ax1.imshow(img_samp_color);
ax1.set_title('Original Colored Image ')
ax2.imshow(img_samp, cmap = 'gray');
ax2.set_title('Chosen Seeds on Gray Image')
ax2.plot(markers[:, 1], markers[:, 0], "r.")
ax3.imshow(segmat, cmap = 'gray');
ax3.set_title('Segmentated Image with Region growing ')
plt.show()

