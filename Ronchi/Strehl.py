import cv2
import numpy as np
from scipy.optimize import curve_fit
from skimage import morphology

def read_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image from {file_path}")
    return image

def threshold_image(image, method='global', threshold_value=128):
    if method == 'global':
        _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    elif method == 'adaptive':
        thresholded_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        raise ValueError("Invalid thresholding method. Choose 'global' or 'adaptive'.")
    return thresholded_image

def denoise_image(image, method='gaussian', kernel_size=3):
    if method == 'gaussian':
        denoised_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        denoised_image = cv2.medianBlur(image, kernel_size)
    elif method == 'bilateral':
        denoised_image = cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        raise ValueError("Invalid denoising method. Choose 'gaussian', 'median', or 'bilateral'.")
    return denoised_image

def detect_edges(image, method='canny', low_threshold=50, high_threshold=150):
    if method == 'canny':
        edges = cv2.Canny(image, low_threshold, high_threshold)
    elif method == 'sobel':
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobel_x, sobel_y)
    else:
        raise ValueError("Invalid edge detection method. Choose 'canny' or 'sobel'.")
    return edges

def skeletonize_image(image):
    binary_image = image.astype(bool)
    skeleton = morphology.skeletonize(binary_image)
    return skeleton.astype(np.uint8) * 255

def analyze_fringes(image, aperture_diameter, focal_length, grid_spacing, grid_thickness):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = [np.mean(cnt, axis=0)[0] for cnt in contours]
    
    def linear_func(x, a, b):
        return a * x + b

    x_data = np.array([pt[0] for pt in centroids])
    y_data = np.array([pt[1] for pt in centroids])
    params, _ = curve_fit(linear_func, x_data, y_data)
    
    # TODO: Calculate wavefront error using aperture_diameter, focal_length, grid_spacing, grid_thickness
    # wavefront_error = <Your formula to calculate wavefront error>
    # Placeholder line to be replaced with actual computation:
    wavefront_error = np.mean(np.abs(y_data - linear_func(x_data, *params)))

    return wavefront_error

def calculate_strehl_value(wavefront_error, wavelength):
    strehl_ratio = np.exp(-(2 * np.pi * wavefront_error / wavelength) ** 2)
    return strehl_ratio

def save_image(file_path, image):
    cv2.imwrite(file_path, image)

def main(image_file_path, wavelength, aperture_diameter, focal_length, grid_spacing, grid_thickness):
    image = read_image(image_file_path)
    
    # Preprocess the image
    thresholded_image = threshold_image(image, method='adaptive')
    denoised_image = denoise_image(thresholded_image, method='gaussian', kernel_size=5)
    edges = detect_edges(denoised_image, method='canny', low_threshold=50, high_threshold=150)
    skeleton = skeletonize_image(edges)
    
    save_image("processed_image.png", skeleton)

    # Analyze the fringes and calculate the wavefront error
    wavefront_error = analyze_fringes(skeleton, aperture_diameter, focal_length, grid_spacing, grid_thickness)

    # Calculate the Strehl value
    strehl_value = calculate_strehl_value(wavefront_error, wavelength)

    print(f"Strehl value: {strehl_value}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 7:
        print(f"Usage: {sys.argv[0]} IMAGE_FILE_PATH WAVELENGTH APERTURE_DIAMETER FOCAL_LENGTH GRID_SPACING GRID_THICKNESS")
        sys.exit(1)

    image_file_path = sys.argv[1]
    wavelength = float(sys.argv[2])
    aperture_diameter = float(sys.argv[3])
    focal_length = float(sys.argv[4])
    grid_spacing = float(sys.argv[5])
    grid_thickness = float(sys.argv[6])

    main(image_file_path, wavelength, aperture_diameter, focal_length, grid_spacing, grid_thickness)
