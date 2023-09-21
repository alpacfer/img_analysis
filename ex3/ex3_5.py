from skimage import color
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import time
import cv2
from skimage.filters import threshold_otsu


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)
    # Change size of window
    cv2.resizeWindow(win_name, 500, 400)


def process_gray_image(img):
    proc_img = img.copy()


def process_rgb_image(img):
    """
    Simple processing of a color (RGB) image
    """
    # Copy the image information so we do not change the original image
    proc_img = img.copy()
    r_comp = proc_img[:, :, 0]
    proc_img[:, :, 0] = 1 - r_comp
    return proc_img


def histogram_stretch(img_in):
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0

    # Linear histogram stretching equation
    img_out = (max_desired - min_desired) / (max_val - min_val) * (img_float - min_val) + min_desired

    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)


def threshold_image(img, threshold):
    # Do the thresholding
    img_out = img > threshold
    # Convert to ubyte
    return img_as_ubyte(img_out)


def gamma_map(img, gamma):
    # Convert to float
    img_float = img_as_float(img)
    # Do the gamma mapping
    img_out = img_float ** gamma
    # Convert to ubyte
    return img_as_ubyte(img_out)


def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # To keep track of frames per second using a high-performance counter
    old_time = time.perf_counter()
    fps = 0
    stop = False
    process_rgb = False  # Set to True to process RGB images, False to process gray scale images
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Change from OpenCV BGR to scikit image RGB
        new_image = new_frame[:, :, ::-1]
        # Flip image horizontally
        new_image = cv2.flip(new_image, 1)
        new_image_gray = color.rgb2gray(new_image)

        # Image processing
        # Histogram stretching of the gray scale image
        hist_img = histogram_stretch(new_image_gray)

        # Gamma mapping of the gray scale image
        gamma_img = gamma_map(hist_img, 2)

        # Thresholding of the gamma image
        otsu_threshold = threshold_otsu(hist_img)
        thresh_img = threshold_image(hist_img, otsu_threshold)

        # Display the resulting frame
        show_in_moved_window('Input', hist_img, 0, 0)
        show_in_moved_window('Processed image', thresh_img, 600, 0)
        show_in_moved_window('Gamma image', gamma_img, 0, 410)
        # show_in_moved_window('Processed image', proc_img, 0, 500)

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()
