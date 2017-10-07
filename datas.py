import cv2
from multiprocessing import Pool
import os
import sys


def resize(image):
    path = os.path.dirname(__file__)
    img_path = os.path.join(path, 'data/RGBflowTest/') + 'frame' + str(image) + '.jpg'
    output = os.path.join(path, 'data/resizedtest/')
    im = cv2.imread(img_path)
    height, width = im.shape[:2]
    cv2.imwrite(os.path.join(output, "frame%d.jpg" % image),
                cv2.resize(im, (int(width / 4), int(height / 4)), interpolation=cv2.INTER_AREA))


if __name__ == '__main__':
    p = Pool(4)
    images = [x for x in range(10798)]
    for i, _ in enumerate(p.imap_unordered(resize, images), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / len(images)))
    p.close()
    p.join()
