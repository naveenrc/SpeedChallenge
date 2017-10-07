import cv2
import numpy as np
import os
import pandas as pd
import sys


path = os.path.dirname(__file__)
inp_path = os.path.join(path, 'data/') + 'testmodified.mp4'
out_path = os.path.join(path, 'data/RGBflowTest/') + 'frame'
cap = cv2.VideoCapture(inp_path)
speeds = pd.read_table(os.path.join(path, 'data/') + 'train.txt', delimiter='\n', header=None)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
count = 0
while 1:
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, cv2.CV_32FC2, 0.4, 1, 25, 2, 8, 1.3, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # hsv = np.asarray(hsv, dtype=np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb, "{}, {}".format(hsv[..., 0], hsv[..., 2]),
                (20, 20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame2', rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
    '''
    cv2.imwrite(out_path + str(count) + '.jpg', rgb)
    count += 1
    sys.stderr.write('\rdone {0:}'.format(count))
    prvs = next

cap.release()
cv2.destroyAllWindows()
