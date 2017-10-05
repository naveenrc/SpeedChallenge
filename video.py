import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


if __name__ == '__main__':
    path = os.path.dirname(__file__)
    inp = os.path.join(path, 'data/modified/') + 'frame'
    output = os.path.join(path, 'data/') + 'out.mp4'
    images = [x for x in range(0, 20399)]

    # Determine the width and height from the first image
    image_path = inp + str(images[0]) + '.jpg'
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:

        image_path = inp + str(image) + '.jpg'
        frame = cv2.imread(image_path)

        out.write(frame)  # Write out frame to video

        cv2.imshow('video', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))
