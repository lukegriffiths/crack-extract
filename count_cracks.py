import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from matplotlib import cm

plt.close('all')

""" Plot parameters """
rcParams = matplotlib.rcParams
rcParams['svg.fonttype'] = 'none'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = '8'
rcParams['legend.numpoints'] = 1

""" Import and skeletonise image """
filename = './output/img11.jpg'
img = cv2.imread(filename, 0)  # Load crack image

# img = np.uint8( scipy.misc.imread(filename, flatten = True) )
img[img > 0] = 255
img_orig = img
# img = img[0:2000, 0:2000]

res = np.floor(1000 * 261. / 526)  # Pixels per mm
img_height = img.shape[0]
img_width = img.shape[1]
area = (img_height / res) * (img_width / res)  # Area of image in mm

plt.figure()
plt.imshow(img, cmap='gray', interpolation='none')  # Show image
plt.xticks([]), plt.yticks([])  # To hide tick values on X and Y axis
plt.tight_layout()
plt.show()

""" Build Grids """
large_gridsize_mm = 1  # Large grid size in mm
small_gridsize_mm = 0.1  # Fine grid size in mm
small_gridsize = np.ceil(small_gridsize_mm * res).astype(int)  # Number of pixels to cover small grid element
large_gridsize = np.ceil(small_gridsize * large_gridsize_mm / small_gridsize_mm).astype(int)  # Number of pixels to cover large grid element
grid_hor = np.uint8(np.zeros(img.shape))  # Create table same size as image
grid_ver = np.uint8(np.zeros(img.shape))
grid_hor[::small_gridsize, :] = 255  # Set horizontal lines white
grid_ver[:, ::small_gridsize] = 255  # Set vertical lines white
large_grid_hor = np.uint8(np.zeros(img.shape))  # Create table same size as image
large_grid_ver = np.uint8(np.zeros(img.shape))
large_grid_hor[::large_gridsize, :] = 255  # Set horizontal lines white
large_grid_ver[:, ::large_gridsize] = 255  # Set vertical lines white

""" Plot images with grid overlain"""
small_grid = grid_hor + grid_ver
small_grid[small_grid > 0] = 255
large_grid = large_grid_hor + large_grid_ver
large_grid[large_grid > 0] = 255

filename = './output/img13.png'
img_labelled = matplotlib.image.imread(filename)  # Load crack image

img_labelled[:, :, 0][small_grid == 255] = 150
img_labelled[:, :, 1][small_grid == 255] = 150
img_labelled[:, :, 2][small_grid == 255] = 150
img_labelled[:, :, 0][large_grid == 255] = 1
img_labelled[:, :, 1][large_grid == 255] = 1
img_labelled[:, :, 2][large_grid == 255] = 1

fig = plt.figure(figsize=(9 / 2.54, 5 / 2.54))
ax = fig.add_subplot(1, 1, 1)  # Show image, grids and intersections
plt.imshow(img_labelled, interpolation='nearest')  # Show image
plt.xticks([]), plt.yticks([])  # To hide tick values on X and Y axis
plt.tight_layout()
plt.savefig('./output/test.png')
plt.show()

matplotlib.image.imsave('GGGIntact2_cracks_labelled_grid.png', img_labelled)

""" Calculate crack properties over entire image """

int_hor = np.zeros_like(img)  # Black image
int_hor[np.logical_and(img == 255, grid_hor == 255)] = 255  # Where img and grid interect, color intersections white
int_ver = np.zeros_like(img)  # Black image
int_ver[np.logical_and(img == 255, grid_ver == 255)] = 255
cracks, nb_hor = ndi.label(int_hor)  # Number of horizontal intersections
cracks, nb_ver = ndi.label(int_ver)  # Number of vertical intersections
cracks, nb = ndi.label(img)  # Label all cracks
Na = nb / area  # Cracks per surface
slices = ndi.find_objects(cracks)  # Find the locations of all objects
length = np.zeros([nb, 1])  # Will contain all crack lengths
for k in np.arange(0, len(slices)):  # Go through each cluster
    [crack_width, crack_height] = img[slices[k]].shape
    length[k] = np.sqrt(crack_width ** 2 + crack_height ** 2) / res  # Crack length in mm as if straight

average_length = np.mean(length)  # Calculate mean crack length

PI = nb_hor / ((img_width / res) * (
            img_height / small_gridsize))  # nb. of intersections with horizontal / ( nb. of lines * line length )
PII = nb_ver / ((img_height / res) * (
            img_width / small_gridsize))  # nb. of intersections with vertical lines / ( nb. of lines * line length )
Sv = PI + PII  # Underwood 1980

# Print results
print('### Crack properties over entire image ### \n')
print('nb hor = ' + str(nb_hor) + ', nb_ver = ' + str(nb_ver))
print('Large gridsize = ' + str(large_gridsize_mm) + ' mm')
print('Fine gridsize = ' + str(small_gridsize_mm) + ' mm')
print('PI = ' + str(PI) + ', PII = ' + str(PII) + ', Sv = ' + str(Sv))  # Print results
print('Walsh crack density Na (number/area) = ' + str(Na) + ' mm^-2')
print('Average crack length = ' + str(average_length) + ' mm')
print('Calculated D0 from Sv = ' + str(Sv * np.pi ** 2 * average_length / 32.))
print('Calculated D0 from Na = ' + str(Na * (average_length / 2) ** 2 * np.pi / 2))

""" Calculate crack properties in patches """
large_gridsize_height_pixels = np.ceil(img_height / large_gridsize).astype(int)
large_gridsize_width_pixels = np.ceil(img_width / large_gridsize).astype(int)

crack_density = np.zeros([large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches
crack_length = np.zeros([large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches
Na = np.zeros([large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches
PI = np.zeros([large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches
PII = np.zeros([large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches

""" Compare image and grid """
for i in range(crack_density.shape[0]):
    for j in range(crack_density.shape[1]):
        iMin = i * (large_gridsize)
        iMax = min((i + 1) * large_gridsize, img.shape[0])
        jMin = j * (large_gridsize)
        jMax = min((j + 1) * large_gridsize, img.shape[1])
        img_section = img[iMin:iMax, jMin:jMax]
        int_hor = np.zeros_like(img_section)  # Black image
        int_hor[np.logical_and(img_section == 255, grid_hor[iMin:iMax,
                                                   jMin:jMax] == 255)] = 255  # Where img and grid interect, color intersections white
        int_ver = np.zeros_like(img_section)  # Black image
        int_ver[np.logical_and(img_section == 255, grid_ver[iMin:iMax, jMin:jMax] == 255)] = 255
        cracks, nb_hor = ndi.label(int_hor)  # Number of horizontal intersections
        cracks, nb_ver = ndi.label(int_ver)  # Number of vertical intersections
        cracks, nb = ndi.label(img_section)  # Number of cracks in patch
        height = img_section.shape[0]
        width = img_section.shape[1]
        area = (height / res) * (width / res)
        Na[i, j] = nb / area
        slices = ndi.find_objects(cracks)  # Find the locations of all objects
        length = np.zeros([nb, 1])
        for k in np.arange(0, len(slices)):  # Go through each cluster
            [crack_width, crack_height] = img_section[slices[k]].shape
            length[k] = np.sqrt(crack_width ** 2 + crack_height ** 2)
        #       plt.figure() # Show image, grids and intersections
        #       plt.imshow(int_ver + int_hor,  interpolation='nearest') # Show image
        #       plt.xticks([]), plt.yticks([])  # To hide tick values on X and Y axis
        #       plt.tight_layout()
        #       plt.show()
        crack_length[i, j] = np.mean(length) / res
        PI[i, j] = nb_hor / ((width / res) * (height / small_gridsize))
        PII[i, j] = nb_ver / ((height / res) * (width / small_gridsize))

Sv = PI + PII

print('\n \n ### Average crack properties over patched image ### \n')
print('nb hor = ' + str(np.mean(nb_hor)) + ', nb_ver = ' + str(np.mean(nb_ver)))
print('Large gridsize = ' + str(large_gridsize_mm) + ' mm')
print('Fine grid size = ' + str(small_gridsize_mm) + ' mm')
print('PI = ' + str(np.mean(PI)) + ', PII = ' + str(np.mean(PII)) + ', Sv = ' + str(np.mean(Sv)))  # Print results
print('Walsh crack density Na (number/area) = ' + str(np.mean(Na)) + ' mm^-2')
print('Average crack length = ' + str(np.mean(crack_length)) + ' mm')
print('Calculated D0 from Sv = ' + str(np.mean(Sv) * np.pi ** 2 * np.mean(crack_length) / 32.))
print('Calculated D0 from Na = ' + str(np.mean(Na) * (np.mean(crack_length) / 2) ** 2 * np.pi / 2))

fig = plt.figure(figsize=(8 / 2.54, 7 / 2.54))  # Show image, grids and intersections
ax = plt.imshow(Sv, extent=[0, img_width / res, 0, img_height / res], cmap=cm.YlOrRd)  # Show image
plt.xlabel('mm')
plt.ylabel('mm')
cb = plt.colorbar(ax, orientation='horizontal')
cb.set_label('Sv (mm2/mm3)')
plt.tight_layout()
plt.savefig('./output/Sv.svg', dpi=600)
plt.show()
