# Importing all necessary libraries
import cv2
import os
import numpy as np
from hafman import Huffman

class Decoder:
    frams=[]

    def __init__(self):
        self.block_size=60
        self.get_frames_from_encoded_file()
        self.decode()
        self.endCapture()

    def get_frames_from_encoded_file(self,name=None,directory_path=None):
        if directory_path is not None:
            string_input = open(os.path.join(directory_path, name)).readline()
        else:
            huffman = Huffman(binary_coded_address="colored_coded_frames/huffman_coded.txt", tree_address="colored_coded_frames/tree.txt")
            string_input = huffman.decode()
        all_frames = []
        for x in string_input.split(' '):
            try:
                all_frames.append(int(x))
            except:
                pass
        end=all_frames[4]+5
        start=0
        while end<len(all_frames):
            frame=all_frames[start:end]
            new_frame=frame[5:end]
            new_frame.extend(frame[:4])
            start=end
            end=all_frames[end+4]+5+end
            print(all_frames[start])
            self.frams.append(new_frame)

    def endCapture(self):
        # Release all space and windows once done
        self.decoder_output.release()
        cv2.destroyAllWindows()

    def differenceReverse(self,current_frame,prev_frame):
        current_frame = current_frame.astype('int16')
        for i in range(current_frame.shape[0]):
            array1 = np.array(current_frame[i])
            array2 = np.array(prev_frame[i])
            current_frame[i] = np.add(array1, array2)
        return current_frame
    def decode(self):
        new_frames=[]
        prev_frame = []
        for frame in self.frams:
            properties=frame[-4:]
            x=frame[:-4]
            x=np.array(x)
            x=self.runLengthScanReverse(x)
            x=x.reshape((int(properties[0]*properties[1]/(self.block_size * self.block_size)),self.block_size *self.block_size))
            x=self.zigzag_scan_reverse(properties,x,self.block_size)
            x=self.quantization_reverse(x,6)
            x=self.DCTReverse(x) #frame_name=f"frame{properties[2]}"
            print (f"frame{properties[2]}")
            if properties[3] == 1:
                x = self.differenceReverse(x, prev_frame)
            prev_frame = x
            x = x.reshape(x.shape[0], int(x.shape[1] / 3), 3)
            # x = x.astype('uint8')
            new_frames.append(x)
            frame_name = f"frame{properties[2]}"
            filename = "colored_out/" + frame_name + ".jpg"
            cv2.imwrite(filename, x)
        self.createDecodedVideo(len(new_frames))

    def createDecodedVideo(self, numberOfFrames):
        fps = 25
        self.width = 960
        self.height = 540
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.decoder_output = cv2.VideoWriter('decoder_colored_output.avi',self.fourcc, fps, (self.width, self.height), True)
        for i in range(numberOfFrames):
            img = cv2.imread("./colored_out/frame" + str(i) + ".jpg", cv2.IMREAD_COLOR)
            self.decoder_output.write(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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

    def DCTReverse(self, array, B=8):
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
        Trans=np.array(Trans,np.int)
        return Trans

if __name__ == "__main__":
    decoder = Decoder()