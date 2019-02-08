import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from matplotlib import cm

""" Plot parameters """
rcParams = matplotlib.rcParams
rcParams['svg.fonttype'] = 'none'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = '8'
rcParams['legend.numpoints'] = 1


def calculate_gridsize_in_pixels(small_gridsize_mm, large_gridsize_mm, resolution):
    """
    Calculate the size in pixels of the small and large grids from the desired sizes in mm, and the resolution

    Args:
        small_gridsize_mm: small spacing of gridlines in millimeters
        large_gridsize_mm: large spacing of gridlines in millimeters
        resolution: resolution of the image in pixel/mm

    Returns:
        small_gridsize (int): small spacing of gridlines in pixels
        large_gridsize (int): large spacing of gridlines in pixels
    """

    small_gridsize = np.ceil(small_gridsize_mm * resolution).astype(int)
    large_gridsize = np.ceil(small_gridsize * large_gridsize_mm / small_gridsize_mm).astype(int)

    return small_gridsize, large_gridsize


def make_grid(small_gridsize, large_gridsize, image_shape, resolution):
    """
    Build grids in which to calculate crack length and spatial density

    small_gridsize_mm:
    :param large_gridsize_mm:
    :param resolution:
    :return:

    """
    grid_hor = np.uint8(np.zeros(image_shape))  # Create table same size as image
    grid_ver = np.uint8(np.zeros(image_shape))
    grid_hor[::small_gridsize, :] = 255  # Set horizontal lines to white
    grid_ver[:, ::small_gridsize] = 255  # Set vertical lines to white
    large_grid_hor = np.uint8(np.zeros(image_shape))  # Create table same size as image
    large_grid_ver = np.uint8(np.zeros(image_shape))
    large_grid_hor[::large_gridsize, :] = 255  # Set horizontal lines to  white
    large_grid_ver[:, ::large_gridsize] = 255  # Set vertical lines to white

    # Create grid
    small_grid = grid_hor + grid_ver
    small_grid[small_grid > 0] = 255
    large_grid = large_grid_hor + large_grid_ver
    large_grid[large_grid > 0] = 255

    return small_grid, large_grid, grid_hor, grid_ver


def crack_statistics_in_whole_image(img, grid_hor, grid_ver, area, small_gridsize, img_height, img_width, resolution):
    """
    Calculate crack properties averaged across entire image and prints them to console

    Args:
        img
        grid_hor:
        grid_ver:
        area:
        small_gridsize_img_height:
        img_width:
        resolution:

    Returns:


    """
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
        length[k] = np.sqrt(crack_width ** 2 + crack_height ** 2) / resolution  # Crack length in mm as if straight

    average_length = np.mean(length)  # Calculate mean crack length

    PI = nb_hor / ((img_width / resolution) * (
            img_height / small_gridsize))  # nb. of intersections with horizontal / ( nb. of lines * line length )
    PII = nb_ver / ((img_height / resolution) * (
            img_width / small_gridsize))  # nb. of intersections with vertical lines / ( nb. of lines * line length )
    Sv = PI + PII  # Underwood 1980

    # Print results
    print('### Crack properties over entire image ### \n')
    print('nb hor = ' + str(nb_hor) + ', nb_ver = ' + str(nb_ver))
    print('PI = ' + str(PI) + ', PII = ' + str(PII) + ', Sv = ' + str(Sv))  # Print results
    print('Walsh crack density Na (number/area) = ' + str(Na) + ' mm^-2')
    print('Average crack length = ' + str(average_length) + ' mm')
    print('Calculated D0 from Sv = ' + str(Sv * np.pi ** 2 * average_length / 32.))
    print('Calculated D0 from Na = ' + str(Na * (average_length / 2) ** 2 * np.pi / 2))


def crack_statistics_in_windows(img, img_height, img_width, large_gridsize, grid_hor, grid_ver, small_gridsize,
                                resolution):
    large_gridsize_height_pixels = np.ceil(img_height / large_gridsize).astype(int)
    large_gridsize_width_pixels = np.ceil(img_width / large_gridsize).astype(int)

    crack_density = np.zeros(
        [large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches
    crack_length = np.zeros(
        [large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches
    Na = np.zeros([large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches
    PI = np.zeros([large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches
    PII = np.zeros([large_gridsize_height_pixels, large_gridsize_width_pixels])  # Nb. of elements = nb. of patches

    # Compare image and grid
    for i in range(crack_density.shape[0]):
        for j in range(crack_density.shape[1]):
            iMin = i * (large_gridsize)
            iMax = min((i + 1) * large_gridsize, img.shape[0])
            jMin = j * (large_gridsize)
            jMax = min((j + 1) * large_gridsize, img.shape[1])
            img_section = img[iMin:iMax, jMin:jMax]
            int_hor = np.zeros_like(img_section)  # Black image
            # Where img and grid intersect, color intersections white
            int_hor[np.logical_and(img_section == 255, grid_hor[iMin:iMax, jMin:jMax] == 255)] = 255

            int_ver = np.zeros_like(img_section)  # Black image
            int_ver[np.logical_and(img_section == 255, grid_ver[iMin:iMax, jMin:jMax] == 255)] = 255
            cracks, nb_hor = ndi.label(int_hor)  # Number of horizontal intersections
            cracks, nb_ver = ndi.label(int_ver)  # Number of vertical intersections
            cracks, nb = ndi.label(img_section)  # Number of cracks in patch
            height = img_section.shape[0]
            width = img_section.shape[1]
            area = (height / resolution) * (width / resolution)
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
            crack_length[i, j] = np.mean(length) / resolution
            PI[i, j] = nb_hor / ((width / resolution) * (height / small_gridsize))
            PII[i, j] = nb_ver / ((height / resolution) * (width / small_gridsize))

        Sv = PI + PII

        # Print microcrack statistics to console
        print('\n \n ### Average crack properties over patched image ### \n')
        print('nb hor = ' + str(np.mean(nb_hor)) + ', nb_ver = ' + str(np.mean(nb_ver)))
        print('PI = ' + str(np.mean(PI)) + ', PII = ' + str(np.mean(PII)) + ', Sv = ' + str(np.mean(Sv)))
        print('Walsh crack density Na (number/area) = ' + str(np.mean(Na)) + ' mm^-2')
        print('Average crack length = ' + str(np.mean(crack_length)) + ' mm')
        print('Calculated D0 from Sv = ' + str(np.mean(Sv) * np.pi ** 2 * np.mean(crack_length) / 32.))
        print('Calculated D0 from Na = ' + str(np.mean(Na) * (np.mean(crack_length) / 2) ** 2 * np.pi / 2))

    return Sv, PI, PII, Na, crack_length


def calculate_crack_statistics(filepath, resolution, large_gridsize_mm=1, small_gridsize_mm=0.1):
    """
    Args:
    filepath (string): filepath of binary image containing pixel-width cracks
    resolution (int): resolution of image in pixels per mm
    large_gridsize_mm (float): size of large grid in mm (grid within which image statistics are calculated)
    small_gridsize_mm (float): size of small grid in mm (grid with which intersections with cracks are counted)

    Returns:
    Sv (NumPy array): crack surface per unit volume (see Underwood (1979)) = PI + PII
    PI (NumPy array): number of line intersections of cracks in horizontal direction per mm
    PII (NumPy array): number of line intersections of cracks in vertical direction per mm
    Na (NumPy array): Number of cracks per unit area (mm x mm)
    crack_length (NumPy array): average crack length
    """
    # Load skeletonised crack image
    img = cv2.imread(filepath, 0)

    # Make sure image is binary (0 = background, 255 = cracks)
    img[img > 0] = 255

    # image dimensions
    img_height = img.shape[0]
    img_width = img.shape[1]
    area = (img_height / resolution) * (img_width / resolution)  # Area of image in mm

    # Plot segmented crack image
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='none')  # Show image
    plt.xticks([]), plt.yticks([])  # To hide tick values on X and Y axis
    plt.tight_layout()
    plt.show()

    small_gridsize, large_gridsize = calculate_gridsize_in_pixels(small_gridsize_mm, large_gridsize_mm, resolution)

    small_grid, large_grid, grid_hor, grid_ver = make_grid(small_gridsize, large_gridsize, img.shape, resolution)

    crack_statistics_in_whole_image(img, grid_hor, grid_ver, area, small_gridsize, img_height, img_width, resolution)

    Sv, PI, PII, Na, crack_length = crack_statistics_in_windows(img, img_height, img_width, large_gridsize, grid_hor,
                                                                grid_ver, small_gridsize, resolution)

    return Sv, PI, PII, Na, crack_length


def plot_crack_statistics(X, img_width, img_height, resolution, colorbar_label=None, output_filepath=None):
    plt.figure(figsize=(8 / 2.54, 7 / 2.54))  # Show image, grids and intersections
    ax = plt.imshow(X, extent=[0, img_width / resolution, 0, img_height / resolution], cmap=cm.YlOrRd)  # Show image
    plt.xlabel('mm')
    plt.ylabel('mm')
    cb = plt.colorbar(ax, orientation='horizontal')
    if colorbar_label is not None:
        cb.set_label(colorbar_label)
    plt.tight_layout()
    if output_filepath is not None:
        plt.savefig(output_filepath, dpi=600)
    plt.show()


def plot_image_with_grid_overlay(filepath, small_grid, large_grid, output_filepath=None):
    """
    Args:
    filepath (string): filepath of image to plot grid over
    small_grid (NumPy array): small grid array
    large_grid (NumPy array): large grid array
    output_filepath (string): filepath of output image of graph (can be .png, .svg ...)
    """
    # filename = './output/img13.png'
    img_labelled = matplotlib.image.imread(filepath)  # Load crack image

    # Overlay grid on labelled crack image
    img_labelled[:, :, 0][small_grid == 255] = 150
    img_labelled[:, :, 1][small_grid == 255] = 150
    img_labelled[:, :, 2][small_grid == 255] = 150
    img_labelled[:, :, 0][large_grid == 255] = 1
    img_labelled[:, :, 1][large_grid == 255] = 1
    img_labelled[:, :, 2][large_grid == 255] = 1

    # Plot
    fig = plt.figure(figsize=(9 / 2.54, 5 / 2.54))
    fig.add_subplot(1, 1, 1)  # Show image, grids and intersections
    plt.imshow(img_labelled, interpolation='nearest')  # Show image
    plt.xticks([]), plt.yticks([])  # To hide tick values on X and Y axis
    plt.tight_layout()
    if output_filepath is not None:
        plt.savefig(output_filepath, dpi=600)
    plt.show()
