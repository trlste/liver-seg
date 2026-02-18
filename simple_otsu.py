import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_multiotsu
import skimage.io as skio
from skimage.color import label2rgb, rgb2lab, lab2rgb, rgb2gray, rgb2hed
from skimage import img_as_ubyte
from skimage.restoration import inpaint

filename = sys.argv[1]
image = skio.imread(filename)
image = image[:, :, :3]

def trim_img(raw_image):
   # tolerance for channel equality
   tol = 2   # use 2 for uint8 images

   # Compute per-pixel color difference
   # If max difference between channels is small → grayscale pixel
   diff = np.max(raw_image, axis=2) - np.min(raw_image, axis=2)

   # Boolean mask: True where pixel is grayscale
   gray_pixel = diff <= tol

   # A column is removable if ALL pixels in that column are grayscale
   gray_column = np.all(gray_pixel, axis=0)

   # Keep columns that are NOT fully grayscale
   valid_cols = np.where(~gray_column)[0]

   if len(valid_cols) > 0:
      image_cropped = raw_image[:, :valid_cols[-1] + 1]
   else:
      image_cropped = raw_image
   return image_cropped

image = trim_img(image)

#ignore blue/nuclei
# Option 1: remove blue and average out red/green
# gray = np.mean(image[:, :, :2], axis=2)

# Option 2: use rgb2lab
#assert image.dtype == 'uint8'
#image[:, :, 2] = 0
#gray = rgb2lab(image[:, :, :3]) 
#b_channel = gray[:, :, 2]
#blue_mask = b_channel < 0
#gray = inpaint.inpaint_biharmonic(gray, blue_mask, channel_axis=-1)
# gray = rgb2gray(np.clip(gray, a_min=0, a_max=1))
#gray[:, :, 2] = np.clip(gray[:, :, 2], a_min=0, a_max=None)
#gray = np.mean(gray, axis=2)

#Option 3: use rgb2hed
gray = rgb2hed(image)
gray = image[:, :, 2]

thresholds = threshold_multiotsu(gray, classes=3)
regions = np.digitize(gray, bins=thresholds)

print("Thresholds:", thresholds)
print("Unique region labels:", np.unique(regions))
print("Pixel counts:", np.bincount(regions.ravel()))

np.save(filename + '_thresholds.npy', thresholds)
np.save(filename + '_regions.npy', regions)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

# Plotting the original image.
ax[0].imshow(image)
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