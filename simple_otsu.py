import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_multiotsu
import skimage.io as skio
from skimage.color import label2rgb, rgb2lab, lab2rgb, rgb2gray, rgb2hed
from skimage import img_as_ubyte
from skimage.restoration import inpaint
from scipy.stats import norm, gamma, lognorm
from scipy.stats import gaussian_kde


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

max_channel_value = np.max(image)

black_mask = (image == [0, 0, 0]).all(axis=2)

# Replace black pixels with the maximum color
image[black_mask] = max_channel_value

print("trimmed img")
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
hed = rgb2hed(image)
gray = hed[:, :, 2]
print("converted image to hed")
print(np.max(gray), np.min(gray))
thresholds = threshold_multiotsu(gray, classes=3)
regions = np.digitize(gray, bins=thresholds)

print("Thresholds:", thresholds)
print("Unique region labels:", np.unique(regions))
print("Pixel counts:", np.bincount(regions.ravel()))

np.save(filename + '_thresholds.npy', thresholds)
np.save(filename + '_regions.npy', regions)


d_pixels = gray.ravel()
region_labels = regions.ravel()

fig_hist, ax_hist = plt.subplots(figsize=(10, 5))

class_names = ['Background', 'Low DAB', 'High DAB']
colors = ['green', 'red', 'blue']

global_min = d_pixels.min()
global_max = d_pixels.max()

bins = np.linspace(global_min, global_max, 201)
x_eval = np.linspace(global_min, global_max, 500)  

for class_idx in range(3):
    mask = region_labels == class_idx
    class_pixels = d_pixels[mask]

    ax_hist.hist(
        class_pixels,
        bins=bins,
        density=True,              # normalized
        alpha=0.4,                 # transparency so they overlap
        color=colors[class_idx],
        label=class_names[class_idx]
    )
    # ---- KDE ----
    kde = gaussian_kde(class_pixels, bw_method=0.3)
    ax_hist.plot(
        x_eval,
        kde(x_eval),
        color=colors[class_idx],
        linewidth=2.5,
        label=f"{class_names[class_idx]} KDE"
    )
    

ax_hist.set_xlabel('DAB Intensity')
ax_hist.set_ylabel('Probability Density')
ax_hist.set_title('DAB Intensity Distribution by Otsu Class')
ax_hist.legend()

# Add threshold lines
for t in thresholds:
    ax_hist.axvline(x=t, color='black', linestyle='--', alpha=0.5)

fig_hist.savefig(filename + '_dab_histogram.png', dpi=150, bbox_inches='tight')

# Plotting the Multi Otsu result.
my_colors = ['green', 'red', 'blue']
overlay = label2rgb(regions, image=gray, colors=my_colors, alpha=1, bg_label=-1)

# Convert to uint8
overlay_uint8 = img_as_ubyte(overlay)

# Save as TIFF
output_name = filename + '_otsu.tif'
skio.imsave(output_name, overlay_uint8)