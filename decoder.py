# Importing all necessary libraries
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class Decoder:
    frams=[]
    def __init__(self, directory_path):
        names= [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))]
        for name in names:
            file1=open(os.path.join(directory_path, name))
            frame=[int(x) for x in file1.readline().split(" ")]
            self.frams.append(frame)
            self.decode()

    def endCapture(self):
        # Release all space and windows once done
        self.dct_out.release()
        self.video.release()
        cv2.destroyAllWindows()

    def decode(self):
        for frame in self.frams:
            properties=frame[-3:]
            frame=frame[:-3]
            frame=np.array(frame)
            frame=self.runLengthScanReverse(frame)
            frame.reshape((int(properties[0]*properties[1]/3600),3600))
    def zigzagScanReverse(self):
        pass

    def runLengthScanReverse(self,frame):
        counter=0
        new_frame=[]
        for i,row in enumerate(frame):
            if i%2==0:
                counter =row
            if i%2 ==1:
                new_frame.extend(np.zeros(counter))
                new_frame.append(row)
        return np.array(new_frame)




    def quantizationReverse(self):

        pass

    def DCTReverse(self):

        pass
decoder=Decoder("./coded_frames")