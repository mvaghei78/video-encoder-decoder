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
        self.block_size=60
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
        new_frames=[]
        for frame in self.frams:
            properties=frame[-3:]
            frame=frame[:-3]
            frame=np.array(frame)
            frame=self.runLengthScanReverse(frame)
            frame=frame.reshape((int(properties[0]*properties[1]/(self.block_size * self.block_size)),self.block_size *self.block_size))
            frame=self.zigzag_scan_reverse(properties,frame,self.block_size)
            frame=self.quantization_reverse(frame,2)
            self.DCTReverse(frame)
            new_frames.append(self.zigzag_scan_reverse(properties,frame,self.block_size))


    def zig_zag_index(self, k, n):
        # upper side of interval
        if k >= n * (n + 1) // 2:
            i, j = self.zig_zag_index(n * n - 1 - k, n)
            return n - 1 - i, n - 1 - j
        # lower side of interval
        i = int((np.sqrt(1 + 8 * k) - 1) / 2)
        j = k - i * (i + 1) // 2
        return (j, i - j) if i & 1 else (i - j, j)

    def zigzag_60_60(self, frame, block_size):
        M = np.zeros((block_size,block_size), dtype=float)
        for k in range(block_size * block_size):
            (x, y) = self.zig_zag_index(k, block_size)
            M[x, y]=frame[k]
        return M
    def zigzag_scan_reverse(self,properties,frame,block_size):
        new_frame=np.zeros((properties[1],properties[0]))
        i=0
        j=0
        for row in frame:
            block =self.zigzag_60_60(row,block_size)
            new_frame[i:i + block_size,j:j + block_size]=block
            j+=block_size
            if j>=properties[0]-1:
                j=0
                i+=block_size
        return new_frame

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

    def quantization_reverse(self, array, quantize_coeff=4):
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                int_trans = int(array[row][column])
                # if number is negative -> 1.positive it 2.shift it 3.negative it
                if (int_trans < 0):
                    int_trans = abs(int_trans)
                    int_trans = int_trans << quantize_coeff
                    array[row][column] = int_trans * -1
                else:
                    int_trans = int_trans << quantize_coeff
                    array[row][column] = int_trans

        return array

    def DCTReverse(self,array,frame_name="current",B=8):
        h, w = array.shape
        h = int(h)
        w = int(w)
        blocksV = int(h / B)
        blocksH = int(w / B)
        vis0 = np.zeros((h, w), np.float32)
        Trans = np.zeros((h, w), np.float32)
        vis0[:h, :w] = array
        # dct on each 8*8 block
        for row in range(blocksV):
            for col in range(blocksH):
                Trans[row * B:(row + 1) * B, col * B:(col + 1) * B] = cv2.idct(
                    vis0[row * B:(row + 1) * B, col * B:(col + 1) * B])
        cv2.imshow('Live', Trans)
        cv2.waitKey(0)

decoder=Decoder("./coded_frames")