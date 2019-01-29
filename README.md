# Crack extract

Python routine using Open CV2 and Mahotas image processing libraries to extract a binary image cracks (lines of a single pixel in thickness) from a greyscale image. This is the code for the automated microcrack analysis and quantification presented in Griffiths et al. (2017) and the corresponding supplementary materials--both are provided in /References.

Griffiths, L., Heap, M. J., Baud, P., & Schmittbuhl, J. (2017). Quantification of microcrack characteristics and implications for stiffness and strength of granite. International Journal of Rock Mechanics and Mining Sciences, 100, 138–150. https://doi.org/10.1016/j.ijrmms.2017.10.013

## Getting started

First download or clone this folder.

### Prerequisites

Make sure the following are installed: 

* mahotas
* open cv2
* matplotlib
* numpy
* scipy ndimage
* skimage
* PIL (Pillow)

### Example

Run the example:

'example.py' demonstrates the image processing on an example file located at '/input/example_micrograph.png', using the default parameters.

You can then adapt this file to run image files of your choosing, adjusting the parameters accordingly. The signification of these parameters, and guidelines for choosing them, are provided in the documentation of extract_cracks.py and in Griffiths et al. 2017 and supplementary materials, in /References.

