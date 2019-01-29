import numpy as np
import extract_cracks
import matplotlib.pyplot as plt

cmap = plt.cm.YlOrRd  # colormap for crack lengths
cmap.set_bad(color='black')  # set nan colour to black


# Example image file path
filename = './input/GGGIntact2_orig.png'
filename = './input/B-2.tif'

# process image
final = extract_cracks.processImage(filename,
                                    median_filter_size=18,
                                    small_object_size=30,
                                    fill_small_holes_n_iterations=2,
                                    n_prune=15,
                                    bg_greyscale=250,
                                    crack_greyscale=245)

# Resolution of image in pixels per mm
res = np.floor(1000 * 261. / 526)
# image height in pixels
img_height = final.shape[0]
# image width in pixels
img_width = final.shape[1]

# calculate and order by crack length
crack_length = extract_cracks.orderByCrackLength(final, res)  # Calculate crack lengths

# save image
plt.imsave('./output/img13.jpg', crack_length, cmap=cmap)
crack_length = np.ma.masked_where(crack_length == 0, crack_length)  # Mask background

# plot figure of
fig = plt.figure(figsize=(8 / 2.54, 8 / 2.54))  # Show image
res = np.floor(1000 * 261. / 526)  # Pixels per mm
ax = plt.imshow(crack_length, extent=[0, img_width / res, 0, img_height / res], cmap=cmap)  # Show image
plt.xlabel('mm')
plt.ylabel('mm')
cb = plt.colorbar(ax, orientation='horizontal')
cb.set_label('Crack length (mm)')
plt.tight_layout()
plt.savefig('./output/crack_length.png', dpi=600)
plt.show()

# Image and average deviation angle
cut = extract_cracks.RDP(final, 70)

# Calculate crack lengths
crack_length_cut = extract_cracks.orderByCrackLength(cut, res)
#  Mask background
crack_length_cut = np.ma.masked_where(crack_length_cut == 0, crack_length_cut)

# mpl.image.imsave('img13.png', crack_length_cut, cmap=cmap)
extract_cracks.save_image(crack_length_cut, cmap=cmap)

# make figure showing crack lengths
fig = plt.figure(figsize=(8 / 2.54, 8 / 2.54))  # Show image
res = np.floor(1000 * 261. / 526)  # Pixels per mm
img_height = crack_length_cut.shape[0]
img_width = crack_length_cut.shape[1]
ax = plt.imshow(crack_length_cut, extent=[0, img_width / res, 0, img_height / res], cmap=cmap)  # Show image
plt.xlabel('mm')
plt.ylabel('mm')
cb = plt.colorbar(ax, orientation='horizontal')
cb.set_label('Crack length (mm)')
plt.tight_layout()
plt.savefig('./output/crack_length_cut.png', dpi=600)
plt.show()
