import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_multiotsu
import skimage.io as skio
from skimage.color import label2rgb
from skimage import img_as_ubyte

filename = sys.argv[1]
image = skio.imread(filename)
print(image.shape)

#ignore blue/nuclei
gray = np.mean(image[:, :, :2], axis=2)

thresholds = threshold_multiotsu(gray, classes=3)
print(thresholds)
regions = np.digitize(gray, bins=thresholds)

print("Thresholds:", thresholds)
print("Unique region labels:", np.unique(regions))
print("Pixel counts:", np.bincount(regions.ravel()))

np.save(filename + '_thresholds.npy', thresholds)
np.save(filename + '_regions.npy', regions)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

# Plotting the original image.
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
ax[1].hist(gray.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thresholds:
   ax[1].axvline(thresh, color='r')

# Plotting the Multi Otsu result.
my_colors = ['green', 'red', 'blue']
overlay = label2rgb(regions, image=gray, colors=my_colors, alpha=1, bg_label=-1)
ax[2].imshow(overlay)
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')

# Convert to uint8
overlay_uint8 = img_as_ubyte(overlay)

# Save as TIFF
output_name = filename + '_otsu.tif'
skio.imsave(output_name, overlay_uint8)

plt.subplots_adjust()

plt.show()