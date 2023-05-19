**Ronchi Strehl Value Calculator**

I am currently having issues calculating the correct value. Currently I am stuck and with my limited image processing experience I do not know where to go from here. Any help would be appreciatied.

This script is designed to perform image processing and analysis on an input image. It provides several functions that can be used individually or collectively to process images, detect fringes, calculate wavefront errors, and determine Strehl values. This documentation will guide you through the different sections of the code and explain how to use it effectively.

To use this script, you must install the following libraries:

    OpenCV (cv2)
    NumPy (np)
    SciPy (curve_fit)
    scikit-image (morphology)

**Code Sections**

**Section 1: Importing Required Libraries**

**Section 2: Reading an Image**

The read_image function is provided to read an image from a file. It takes a file path as input and returns the image in grayscale mode. If the image fails to be read, a ValueError is raised.

**Section 3: Thresholding an Image**

The threshold_image function performs image thresholding. It takes an image as input along with optional parameters to control the thresholding method and threshold value. The function supports two thresholding methods: global and adaptive. The result is a binary image with pixels set to 255 (white) or 0 (black).

**Section 4: Denoising an Image**

The denoise_image function applies denoising to an image. It takes an image as input along with optional parameters to select the denoising method and kernel size. The supported denoising methods are Gaussian, median, and bilateral. The function returns a denoised image.

**Section 5: Detecting Edges**

The detect_edges function performs edge detection on an image. It takes an image as input along with optional parameters to select the edge detection method and threshold values. The supported edge detection methods are Canny and Sobel. The function returns an image with detected edges.

**Section 6: Skeletonizing an Image**

The skeletonize_image function skeletonizes a binary image. It takes a binary image as input and returns a skeletonized version of the image.

**Section 7: Analyzing Fringes and Calculating Wavefront Error**

The analyze_fringes function analyzes the fringes in an image and calculates the wavefront error. It takes a binary image as input and uses curve fitting to fit a linear function to the fringes. The wavefront error is then calculated as the mean absolute difference between the actual y-coordinates and the values predicted by the linear function.

**Section 8: Calculating Strehl Value**

The calculate_strehl_value function calculates the Strehl value based on the wavefront error and the specified wavelength. The Strehl ratio is calculated using the formula: exp(-(2 * pi * wavefront_error / wavelength)^2).

**Section 9: Saving an Image**

The save_image function saves an image to a file. It takes a file path and an image as input and writes the image to the specified file.

**Section 10: Main Function**

The main function serves as the entry point for the script. It takes the image file path and wavelength as command-line arguments and performs the following steps:

    Reads the input image using the read_image function.
    Preprocesses the image by applying thresholding, denoising, edge detection, and skeletonization.
    Saves the processed image to a file.
    Analyzes the fringes in the skeleton image and calculates the wavefront error using the analyze_fringes function.
    Calculates the Strehl value based on the wavefront error and wavelength using the calculate_strehl_value function.
    Prints the calculated Strehl value.

**Section 11: Command-Line Usage**

The script allows command-line usage for easy execution. It checks if the script is being run as the main program and expects two command-line arguments: the image file path and the wavelength. If the expected number of command-line arguments is not provided, the script displays usage information and exits.

To run the script from the command line, use the following format:

python script.py IMAGE_FILE_PATH WAVELENGTH

Replace script.py with the actual filename of the script, IMAGE_FILE_PATH with the path to the input image file, and WAVELENGTH with the wavelength value for calculating the Strehl value.
