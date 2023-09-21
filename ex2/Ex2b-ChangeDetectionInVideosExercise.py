import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte


def show_in_moved_window(win_name, img, x, y, width, height):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.resizeWindow(win_name, width, height)
    cv2.imshow(win_name, img)


def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = img_as_float(frame_gray)

    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray)

        # Threshold the difference image
        threshold = 0.1
        thresh_img = np.zeros(dif_img.shape)
        thresh_img[dif_img < threshold] = 0
        thresh_img[dif_img >= threshold] = 1

        # Number of foreground pixels
        n_fg_pixels = np.sum(thresh_img)  # Number of foreground pixels

        # Convert to 8-bit image
        thresh_img = img_as_ubyte(thresh_img)

        # Percentage of foreground pixels compared to the total number of pixels in the image
        n_pixels = thresh_img.shape[0] * thresh_img.shape[1]  # Number of pixels in the image
        perc_fg_pixels = n_fg_pixels / n_pixels  # Percentage of foreground pixels
        print(f"Percentage of foreground pixels: {perc_fg_pixels * 100:.2f}%")

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Show an alarm if the percentage of foreground pixels is above a threshold
        A = 0.05
        if perc_fg_pixels > A:
            # Display text: Alarm in red
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(new_frame, "Alarm", (100, 300), font, 3, (0, 0, 255), 2)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Put the percentage of foreground pixels on the new_frame
        str_out = f"perc_fg_pixels: {perc_fg_pixels * 100:.2f}%"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 150), font, 1, 255, 1)

        # Update the background image
        alpha = 0.99
        frame_gray = alpha * frame_gray + (1 - alpha) * new_frame_gray

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10, 640, 480)
        show_in_moved_window('Difference image', dif_img, 600, 10, 640, 480)
        show_in_moved_window('Threshold image', thresh_img, 0, 400, 640, 480)
        show_in_moved_window('Background image', frame_gray, 600, 400, 640, 480)

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()
