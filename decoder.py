# Importing all necessary libraries
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class Decoder:
    def __init__(self, videopath):
        # Read the video from specified path
        self.video = cv2.VideoCapture(videopath)
        ret, frame = self.video.read()

        self.height, self.width, self.nchannels = frame.shape
        fps = 25
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.dct_out = cv2.VideoWriter('dct_output.avi', self.fourcc, fps, (self.width, self.height), 0)
        try:
            # creating a folder named dct
            if not os.path.exists('dct'):
                os.makedirs('dct')
        # if not created then raise error
        except OSError:
            print('Error: Creating directory of dct')

    def endCapture(self):
        # Release all space and windows once done
        self.dct_out.release()
        self.video.release()
        cv2.destroyAllWindows()

    def decode(self):
        pass

    def zigzagScanReverse(self):
        # بعد از عکس runlength نوبت عکس zigzag هست
        pass

    def runLengthScanReverse(self):
        # اول عکس run length اعمال میشه
        pass

    def quantizationReverse(self):
        # بعد از عکس اسکن ها نوبت عکس quantization هست
        pass

    def DCTReverse(self):
        # در نهایت عکس dct اعمال میشه که خود dct رو تو encoder من داخل متد encode پیاده کردم
        #و براش متد جدا در نظر نگرفتم.
        pass
