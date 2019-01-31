# %% Imports
import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import watershed, binary_dilation, binary_erosion
from scipy import ndimage as ndi
from skimage import filters
import mahotas as mh
import matplotlib as mpl
from PIL import Image

# %% Parameters
# Color map
cmap = plt.cm.YlOrRd  # colormap for crack lengths
cmap.set_bad(color='black')  # set nan colour to black

""" Plot parameters """
rcParams = mpl.rcParams
rcParams['svg.fonttype'] = 'none'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = '8'
rcParams['legend.numpoints'] = 1


# %% Functions
def branchPoints(img):
    img[img > 0] = 1  # mh works with binary images
    branch_points = np.zeros_like(img)
    bp = []
    bp.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    bp.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
    bp.append(np.array([[2, 1, 2], [1, 1, 1], [0, 0, 0]]))
    bp.append(np.array([[1, 2, 1], [2, 1, 0], [1, 0, 0]]))
    bp.append(np.array([[2, 1, 0], [1, 1, 0], [2, 1, 0]]))
    bp.append(np.array([[1, 0, 0], [2, 1, 0], [1, 2, 1]]))
    bp.append(np.array([[0, 0, 0], [1, 1, 1], [2, 1, 2]]))
    bp.append(np.array([[0, 0, 1], [0, 1, 2], [1, 2, 1]]))
    bp.append(np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]]))
    bp.append(np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]]))
    bp.append(np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]]))
    bp.append(np.array([[0, 1, 0], [1, 1, 2], [0, 2, 1]]))
    bp.append(np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]]))
    bp.append(np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]]))
    bp.append(np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]]))
    bp.append(np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]]))
    bp.append(np.rot90(bp[-1]))  # Rotation of previous array
    bp.append(np.rot90(bp[-1]))
    bp.append(np.rot90(bp[-1]))
    bp.append(np.array([[1, 0, 2], [1, 1, 1], [0, 1, 0]]))
    bp.append(np.rot90(bp[-1]))
    bp.append(np.rot90(bp[-1]))
    bp.append(np.rot90(bp[-1]))
    bp.append(np.array([[0, 1, 2], [1, 1, 1], [1, 0, 0]]))
    bp.append(np.rot90(bp[-1]))
    bp.append(np.rot90(bp[-1]))
    bp.append(np.rot90(bp[-1]))
    bp.append(np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]]))
    bp.append(np.rot90(bp[-1]))
    bp.append(np.rot90(bp[-1]))
    bp.append(np.rot90(bp[-1]))

    for i in np.arange(0, len(bp)):
        branch_points = branch_points + mh.morph.hitmiss(img, bp[i])

    branch_points[branch_points > 1] = 1  # Remove double points
    return branch_points


def endPoints(skel):
    skel[skel > 0] = 1  # mh works with binary images
    endpoint1 = np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]])
    endpoint2 = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]])
    endpoint3 = np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]])
    endpoint4 = np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]])
    endpoint5 = np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]])
    endpoint6 = np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]])
    endpoint7 = np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]])
    endpoint8 = np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])
    ep1 = mh.morph.hitmiss(skel, endpoint1)
    ep2 = mh.morph.hitmiss(skel, endpoint2)
    ep3 = mh.morph.hitmiss(skel, endpoint3)
    ep4 = mh.morph.hitmiss(skel, endpoint4)
    ep5 = mh.morph.hitmiss(skel, endpoint5)
    ep6 = mh.morph.hitmiss(skel, endpoint6)
    ep7 = mh.morph.hitmiss(skel, endpoint7)
    ep8 = mh.morph.hitmiss(skel, endpoint8)
    ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8
    ep[ep > 1] = 1  # Remove double points
    return ep


def removeSmallObjects(img, size):
    label_objects, nb_labels = ndi.label(img, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    slices = ndi.find_objects(label_objects)  # Find the locations of all objects
    sizes = np.zeros([nb_labels + 1, 1])
    for i in np.arange(0, len(slices)):  # Go through each cluster
        sizes[i] = np.sqrt(img[slices[i]].shape[0] ** 2 + img[slices[i]].shape[1] ** 2)
    mask_size = np.where(sizes > size)[0] + 1
    max_index = np.zeros([nb_labels + 1, 1], np.uint8)
    max_index[mask_size] = 1
    max_index = np.squeeze(max_index)
    label_objects = max_index[label_objects]
    img = 255 * np.uint8(label_objects)
    return img


def removeEndPoints(skeleton, n):  # Iteratively removes end points n times
    for i in np.arange(0, n):
        ep = endPoints(skeleton)  # Find end points
        skeleton = np.logical_and(skeleton, np.logical_not(ep))  # Skeleton without end points
        skeleton = 255 * np.uint8(skeleton)  # Skeleton in standard format
    return skeleton


def removeBranchPoints(skeleton):  # Removes branch points
    branch_points = branchPoints(skeleton)  # Find end points
    skeleton = np.logical_and(skeleton, np.logical_not(branch_points))  # Skeleton without end points
    skeleton = 255 * np.uint8(skeleton)  # Skeleton in standard format
    return skeleton


def restoreBranches(skeleton, skeleton_original):  #
    size_pre = 0
    i = 0
    while size_pre != np.sum(
            skeleton):  # Keeps going until the two consecutive iterations of skeleton have the same sum
        i = i + 1;
        # mpl.image.imsave(str(i) + '_prune.png', skeleton[532:555, 560:595], cmap = 'gray')
        size_pre = np.sum(skeleton)  # Number of white pixels
        end_points_dilated = 255 * np.uint8(binary_dilation(endPoints(skeleton)))  # Dilate end points
        skeleton_dilated = 255 * np.uint8(np.logical_or(skeleton, end_points_dilated))  # Add to pruned image
        # mpl.image.imsave(str(i) + '_prune_dilated.png', skeleton_dilated[532:555, 560:595], cmap = 'gray')
        skeleton = 255 * np.uint8(
            np.logical_and(skeleton_dilated, skeleton_original))  # Intersect dilated and unpruned image
        # mpl.image.imsave(str(i) + '_prune_dilated_masked.png', skeleton[532:555, 560:595], cmap = 'gray')
        bp = branchPoints(skeleton)  # Find branch points in lengthened image
        skeleton = removeBranchPoints(skeleton)  # Remove the branchpoints
        # mpl.image.imsave(str(i) + '_prune_dilated_lone.png', skeleton[532:555, 560:595], cmap = 'gray')
        skeleton = removeSmallObjects(skeleton, 2)  # Remove lone pixels left by branchpoint removal
        # mpl.image.imsave(str(i) + '_prune_nobranch.png', skeleton[532:555, 560:595], cmap = 'gray')
        skeleton = 255 * np.uint8(np.logical_or(skeleton, bp))  # Restore branchpoints
    skeleton = np.uint8(removeBranchPoints(skeleton))  # Keep cracks separate
    skeleton = removeSmallObjects(skeleton, 2)  # Remove lone pixels left by branchpoint removal
    return skeleton


def histEq(img, nbr_bins=256):
    img_hist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)  # Make histogram
    cdf = img_hist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # Normalize by highest value
    img_eq = np.floor(np.interp(img.flatten(), bins[:-1], cdf))  # Interpolate image to new normalised distribution
    return img_eq.reshape(img.shape)


def RDP(img, angle):
    criticalRatio = 2 * np.tan(np.deg2rad(90 - angle / 2))
    size_pre = -1
    new_bp = np.zeros_like(img)  # New branchpoints/crack intersections
    j = 0
    while size_pre != np.sum(new_bp):  # Until no more branchpoints are being made
        j = j + 1
        size_pre = np.sum(new_bp)
        label_objects, nb_labels = ndi.label(img, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # Find all features
        slices = ndi.find_objects(label_objects)  # Find the locations of features
        for i in np.arange(0, len(slices)):  # Go through each clusterdf
            padded_crack = np.lib.pad(label_objects[slices[i]], 1, 'constant',
                                      constant_values=0)  # Pad the slice to be able to detect edge points
            padded_crack[padded_crack != i + 1] = 0  # Remove other cracks in slice
            endpoints = endPoints(padded_crack)  # Find end points
            ep = np.column_stack(np.where(endpoints > 0))  # Find coordinates of end points
            if ep.size > 1:  # if there are at least 2 end points
                line_length = np.sqrt(
                    (ep[0, 0] - ep[1, 0]) ** 2 + (ep[0, 1] - ep[1, 1]) ** 2)  # Calculate length of approx. line
                crack_coords = np.column_stack(np.where(padded_crack > 0))  # Find coordinates of crack
                distance = np.abs(
                    (ep[1, 1] - ep[0, 1]) * crack_coords[:, 0] - (ep[1, 0] - ep[0, 0]) * crack_coords[:, 1] + ep[1, 0] *
                    ep[0, 1] - ep[1, 1] * ep[0, 0]) / line_length
                if np.max(distance) > 0:
                    if line_length / np.max(distance) <= criticalRatio:
                        bp_coords = crack_coords[np.argmax(distance),
                                    :] - 1  # New branch point coords. -1 because crack coords is for padded array
                        bp = np.zeros_like(img[slices[i]])
                        bp[bp_coords[0], bp_coords[1]] = 255
                        new_bp[slices[i]] = bp
        img[new_bp > 0] = 0
    return img


def fillSmallHoles(img, iterations):
    for i in np.arange(0, iterations):
        img = binary_dilation(img)  # Dilate and erode to fill small holes in cracks
    for i in np.arange(0, iterations):
        img = binary_erosion(img)
    return img


def removeEndPointsIter(img):
    label_objects, nb_labels = ndi.label(img, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # Find all cracks
    slices = ndi.find_objects(label_objects)  # Find the locations of all cracks
    new_img = np.zeros_like(label_objects)
    for i in np.arange(0, len(slices)):  # Go through each crack
        crack = copy.copy(label_objects[slices[i]])  # Isolate crack slice
        padded_crack = np.lib.pad(crack, 1, 'constant',
                                  constant_values=0)  # Pad the slice to be able to detect edge points
        padded_crack[padded_crack != i + 1] = 0  # Remove any other cracks in slice
        nb_end_points = np.sum(endPoints(padded_crack))  # Find end points
        while (nb_end_points > 2):  # While the crack has more than 2 end points
            padded_crack = removeEndPoints(padded_crack, 1)  # Remove end points
            nb_end_points = np.sum(endPoints(padded_crack))  # Recalculate number of endpoints
            print(nb_end_points)
        new_img[slices[i]] = 255 * np.uint8(np.logical_or(new_img[slices[i]], padded_crack[1:-1, 1:-1]))
    return new_img


def orderByCrackLength(img, res):  # res in pix/mm
    label_objects, nb_labels = ndi.label(img, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # Find all cracks
    slices = ndi.find_objects(label_objects)  # Find the locations of all objects
    lengths = np.zeros([nb_labels, 1])
    for i in np.arange(0, len(slices)):  # Go through each cluster
        lengths[i] = np.sqrt(img[slices[i]].shape[0] ** 2 + img[slices[i]].shape[1] ** 2)
    for i in np.arange(1, nb_labels + 1):
        label_objects[label_objects == i] = -lengths[i - 1]  # Take negative to overwriting other cracks
    label_objects = -label_objects / res  # FLip back to positive values
    return label_objects


def save_image(img, filepath=None, cmap='gray'):
    save_image.counter += 1  # increment each time function is called

    if filepath is None:
        img = Image.fromarray(np.asarray(np.clip(img, 0, 255),
                                         dtype="uint8"), "L")
        filepath = './output/img' + str(save_image.counter) + '.jpg'
        img.save(filepath)
    else:
        img = Image.fromarray(np.asarray(np.clip(img, 0, 255),
                                         dtype="uint8"), "L")
        img.save(filepath)


def processImage(filepath, median_filter_size=15, small_object_size=40, fill_small_holes_n_iterations=2, n_prune=15,
                 bg_greyscale=250, crack_greyscale=245):
    """
    Function to create extract a binary image pixel-thick cracks from an image of a cracked material.

    For details, see Griffiths et al. (2017) and supplementary materials (provided in ./references folder).

    Griffiths, L., Heap, M.J., Baud, P., Schmittbuhl, J., 2017. Quantification of microcrack characteristics and
    implications for stiffness and strength of granite. International Journal of Rock Mechanics and Mining Sciences 100,
    138–150. https://doi.org/10.1016/j.ijrmms.2017.10.013

    Args:
        filepath (string): path to the image file to be processed
        median_filter_size (int): width (in pixels) of the square median filter window. the value at the center of the
        moving window is replaces by the median of the values within the window. This should be set to a value just
        larger than the crack width.
        small_object_size (int): connected objects within the image following thresholding, with contained within a
        square of this size (in pixels) will be removed from the image.
        fill_small_holes_n_iterations (int): Before skeletonisation of the segmented image, any holes in the segmented
        cracks must be removed. This value gives the number of times the image will be dilated and eroded, to fill all
        holes. 2 is often enough.
        n_prune: Number of times the skeletonised cracks are pruned i.e., end points are removed. This value should be
        greater than the width of the segmented cracks before skeletonisation, or the median_filter_size, which should
        be similar.
        bg_greyscale: The lower boundary of the image mask defined prior to the watershed segmentation. Any pixel with
        a greyscale value below this value is "background"
        crack_greyscale: The lower boundary of the image mask defined prior to the watershed segmentation. Any pixel
        with a greyscale value above this value is "crack".

    Returns:
        cracks_skeleton_restored (numpy array): Binary image of the skeletonised cracks.


    """

    # img_orig = np.uint8( scipy.misc.imread(filename, flatten = True) )
    image = Image.open(filepath).convert('L')
    img_orig = np.uint8(image)
    
    # img = img[2:-2, 2:-2] # Microscope image borders are all same grayscale
    # img = img[760:780, 923:977]

    """ Filter image """
    img = (255 - img_orig)  # Switch grayscale
    img_median = ndi.median_filter(img, median_filter_size)  # Median filter, good at conserving edges
    img_filtered = (255 - cv2.subtract(img, img_median))  # Subtract filtered image from original (cracks = white)
    
    """ Segmentation """
    markers = np.zeros_like(img_filtered)  # Mark the different regions of the image
    markers[img_filtered > bg_greyscale] = 1  # Minimum crack grayscales
    markers[img_filtered < crack_greyscale] = 2  # Maximum crack grayscales

    # Plot median filtered image and mask
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(img_filtered, cmap='gray')
    ax[1].imshow(markers, cmap='gray')
    plt.show()
    
    elevation_map = filters.sobel(img_filtered)  # Edge detection for watershed
    segmented = np.abs(255 - 255 * watershed(elevation_map, markers))  # Watershed segmentation, white on black

    """ Thin, prune and label cracks """
    cracks = removeSmallObjects(segmented, small_object_size)  # Remove small objects
    cracks = fillSmallHoles(cracks, fill_small_holes_n_iterations)  # Fill small holes in cracks
    cracks = binary_dilation(cracks)  # Dilate before thinning
    cracks_skeleton = 255 * np.int8(mh.thin(cracks > 0))  # Skeletonise image
    cracks_skeleton_pruned = removeEndPoints(cracks_skeleton, n_prune)  # Remove skeletonisation artefacts
    cracks_skeleton_pruned_no_bp = removeBranchPoints(cracks_skeleton_pruned)  # Remove branch points to separate cracks
    cracks_skeleton_pruned_no_bp_2_ep = removeEndPointsIter(
        cracks_skeleton_pruned_no_bp)  # Remove end points until 2 per crack
    cracks_skeleton_restored = restoreBranches(cracks_skeleton_pruned_no_bp_2_ep,
                                               cracks_skeleton)  # Restore branches without creating new endpoints

    # Plot original and final image
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(cracks_skeleton_restored, cmap='gray')
    plt.show()
    
    # Save images
    save_image.counter = 0
    save_image(img_orig)
    save_image(255 - img_median)
    save_image(255 - img_filtered)
    
    save_image(255 * elevation_map / np.max(elevation_map))
    save_image(255 * segmented / np.max(segmented))
    save_image(255 * cracks / np.max(cracks))
    save_image(255 * cracks_skeleton / np.max(cracks_skeleton))
    save_image(255 * cracks_skeleton_pruned / np.max(cracks_skeleton_pruned))
    save_image(255 * cracks_skeleton_pruned_no_bp / np.max(cracks_skeleton_pruned_no_bp))
    save_image(cracks_skeleton_pruned_no_bp_2_ep)
    save_image(cracks_skeleton_restored)

    return cracks_skeleton_restored
