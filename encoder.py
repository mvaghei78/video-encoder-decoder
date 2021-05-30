# Importing all necessary libraries
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class Encoder:
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

    def encode(self):
        B = 8  # blocksize
        currentframe = 0
        while (True):
            # reading from frame
            ret, frame = self.video.read()
            if ret:
                B = 8  # blocksize
                img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = np.array(img1.shape[:2]) / B * B
                h = int(h)
                w = int(w)
                img1 = img1[:h, :w]
                blocksV = int(h / B)
                blocksH = int(w / B)
                vis0 = np.zeros((h, w), np.float32)
                Trans = np.zeros((h, w), np.float32)
                vis0[:h, :w] = img1
                for row in range(blocksV):
                    for col in range(blocksH):
                        currentblock = cv2.dct(vis0[row * B:(row + 1) * B, col * B:(col + 1) * B])
                        Trans[row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock

                name = './dct/frame' + str(currentframe) + '.jpg'
                cv2.imwrite(name,Trans)
                currentframe+=1
                # save gray fram in output video file
                self.dct_out.write(Trans)
                # Display the resulting frame
                cv2.imshow('Live', Trans)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    def quantization(self):
        # اول باید quantize بشه
        pass

    def zigzagScan(self):
        # بعد از quantize باید zigzag اسکن بشه
        pass

    def runLengthScan(self):
        # بعد از zizzag اسکن نوبت run length scan هست
        pass
obj = Encoder("sample.mp4")
obj.encode()
obj.endCapture()