import cv2
import os
import matplotlib.image as mpimg
import numpy as np
from multiprocessing import Pool
import sys


def images(in_path, out_path):
    vidcap = cv2.VideoCapture(in_path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        cv2.imwrite(os.path.join(out_path, "frame%d.jpg" % count), image)  # save frame as JPEG file
        count += 1
    print('Total images ' + str(count))


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Eliminates the region of the image defined by the polygon
    formed from `vertices`. Polygon is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    return masked_image


def mask(inp):
    path = os.path.dirname(__file__)
    img_path = os.path.join(path, 'data/testimages/') + 'frame' + str(inp) + '.jpg'
    output = os.path.join(path, 'data/modified/')
    vertices = np.array([[(0, 360), (0, 220), (300, 210), (640, 240),
                          (640, 360), (590, 360), (520, 310), (110, 310),
                          (30, 360)]], dtype=np.int32)
    img = mpimg.imread(img_path)
    masked = region_of_interest(img, vertices)
    cv2.imwrite(os.path.join(output, "frame%d.jpg" % inp), masked)


if __name__ == '__main__':
    
    path = os.path.dirname(__file__)
    inp = os.path.join(path, 'data/') + 'test.mp4'
    output = os.path.join(path, 'data/testimages/')
    images(inp, output)
    
    p = Pool(4)
    images = [x for x in range(0, 10798)]
    for i, _ in enumerate(p.imap_unordered(mask, images), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / len(images)))
    p.close()
    p.join()
