import numpy as np
import extract_cracks
import count_cracks
import matplotlib.pyplot as plt

# Change filepath to use your own image
filepath = './input/example_image.png'
filepath = './input/B-2.tif'

# Resolution of the image in pixels per mm
resolution = np.floor(1000 * 261. / 526)

# Process image
final = extract_cracks.processImage(filepath,
                                    median_filter_size=15,
                                    small_object_size=30,
                                    fill_small_holes_n_iterations=4,
                                    n_prune=20,
                                    bg_greyscale=250,
                                    crack_greyscale=245)

# image height in pixels
img_height = final.shape[0]

# image width in pixels
img_width = final.shape[1]

# crack length colormap
colormap = plt.cm.YlOrRd

# Set NaN color to black
colormap.set_bad(color='black')

# Cracks that have a high curvature are split into smaller cracks
cut = extract_cracks.RDP(final, 70)

# calculate crack lengths from their lengths in pixels and image resolution (pixels/mm) and order by crack length
crack_length_cut = extract_cracks.orderByCrackLength(cut, resolution)

# mask the background to have NaN length
crack_length_cut = np.ma.masked_where(crack_length_cut == 0, crack_length_cut)

# Figure showing cracks, colored according to length
fig = plt.figure(figsize=(8 / 2.54, 8 / 2.54))  # Show image
img_height = crack_length_cut.shape[0]
img_width = crack_length_cut.shape[1]
ax = plt.imshow(crack_length_cut, extent=[0, img_width / resolution, 0, img_height / resolution], cmap=colormap)
plt.xlabel('mm')
plt.ylabel('mm')
cb = plt.colorbar(ax, orientation='horizontal')
cb.set_label('Crack length (mm)')
plt.tight_layout()
plt.savefig('./output/crack_lengths.png', dpi=600)
plt.show()


# Analyse image of cracks
Sv, PI, PII, Na, crack_length = count_cracks.calculate_crack_statistics('./output/img11.jpg', resolution=resolution)

# Plot Sv (crack surface density)
count_cracks.plot_crack_statistics(Sv, img_width, img_height, resolution, colorbar_label='Sv [mm2/mm3]',
                                   output_filepath='./output/Sv.png')

# Plot Na (number of cracks per unit area)
count_cracks.plot_crack_statistics(Na, img_width, img_height, resolution, colorbar_label='Na [/mm2])',
                                   output_filepath='./output/Na.png')

# Plot crack length (endpoint-to-endpoint)
count_cracks.plot_crack_statistics(crack_length, img_width, img_height, resolution, colorbar_label='Crack length [mm])',
                                   output_filepath='./output/crack_length.png')