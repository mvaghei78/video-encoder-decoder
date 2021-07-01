# Importing all necessary libraries
import cv2
import os
import numpy as np
import numpy

from hafman import Huffman


class Encoder:
    def __init__(self, video_path):
        # Read the video from specified path
        self.video = cv2.VideoCapture(video_path)
        try:
            # creating a folder named coded_frames
            if not os.path.exists('coded_frames'):
                os.makedirs('coded_frames')
        # if not created then raise error
        except OSError:
            print('Error: Creating directory of coded_frames')

    def endCapture(self):
        # Release all space and windows once done
        self.video.release()
        cv2.destroyAllWindows()

    def quantization(self, array, quantize_coeff=4):
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                int_trans = int(array[row][column])
                # if number is negative -> 1.positive it 2.shift it 3.negative it
                if ( int_trans < 0 ):
                    int_trans = abs(int_trans)
                    int_trans = int_trans >> quantize_coeff
                    array[row][column] = int_trans * -1
                else:
                    int_trans = int_trans >> quantize_coeff
                    array[row][column] = int_trans

        return array

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
        M = np.zeros((block_size * block_size), dtype='int32')
        for k in range(block_size * block_size):
            (x, y) = self.zig_zag_index(k, block_size)
            M[k] = frame[x, y]
        return M

    def zigzagScan(self, frame, block_size=60):

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        frame_total_size = frame_width * frame_height
        zigzaged_frame = np.zeros((int(frame_total_size / (block_size * block_size)), block_size * block_size),
                                  dtype='int32')
        # how many 60*60 blocks in rows and columns
        block_in_column = int(frame_width / block_size)
        block_in_row = int(frame_height / block_size)
        index = 0
        for row in range(block_in_row):
            for column in range(block_in_column):
                # get each 60*60 block
                slice_frame = frame[row * block_size:(row + 1) * block_size,
                              column * block_size:(column + 1) * block_size]
                # zigzag on 60*60 slice_frame
                zigzaged_frame[index] = self.zigzag_60_60(slice_frame, block_size)
                index += 1

        return zigzaged_frame

    def runLengthScan(self, frame):
        frame = frame.flatten()
        frame = frame.astype(int)
        array = numpy.array([], dtype='int32')
        i = 0
        count = 0
        while i != frame.size:
            if frame[i] == 0:
                count += 1
            else:
                array = np.append(array, [count, frame[i]])
                count = 0
            i += 1
        # if last element is zero then put number of zeros in array
        if frame[i-1] == 0:
            array = np.append(array, [count-1, 0])  # count-1 is number of zeros before zero

        array = array.reshape(int(array.size/2),2)
        return array

    def difference(self,current_frame,prev_frame):
        current_frame = current_frame.astype('int8')
        for i in range(current_frame.shape[0]):
            array1 = np.array(current_frame[i])
            array2 = np.array(prev_frame[i])
            current_frame[i] = np.subtract(array1, array2)
        return current_frame

    def encode(self):
        B = 8  # block size
        currentframe = 0
        all_coded_frames = []
        previous_frame = 0
        frame_type = 0 # 0 -> I , 1 -> P
        while (True):
            # reading from frame
            ret, frame = self.video.read()
            if ret:
                if currentframe % 6 == 0:
                    frame_type = 0
                else:
                    frame_type = 1

                B = 8  # block size
                img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img1_copy = img1
                if frame_type == 1:
                    img1 = self.difference(img1,previous_frame)
                array = np.array(img1.shape[:2])
                h, w = array / B * B
                h = int(h)
                w = int(w)
                blocksV = int(h / B)
                blocksH = int(w / B)
                vis0 = np.zeros((h, w), np.float32)
                Trans = np.zeros((h, w), np.float32)
                vis0[:h, :w] = img1
                # dct on each 8*8 block
                for row in range(blocksV):
                    for col in range(blocksH):
                        Trans[row * B:(row + 1) * B, col * B:(col + 1) * B] = cv2.dct(vis0[row * B:(row + 1) * B, col * B:(col + 1) * B])

                # quantize numbers with shift 6 bit to right
                Trans = self.quantization(Trans,6)
                # zigzag on each 60*60 block
                Trans = self.zigzagScan(Trans, 60)
                # run-length all over frame
                Trans = self.runLengthScan(Trans)
                # output of run-length is 2d numpy array that tell how many zeros are before
                # non-zero numbers
                Trans = Trans.flatten()
                Trans = Trans.astype('int32')
                # append width height and number of frame to last of array
                all_coded_frames.extend(make_frame_array(frame_type, Trans, [w, h, currentframe]))
                Trans = make_frame_array(frame_type, Trans, [w, h, currentframe])
                encoded_file = open("./coded_frames/encoded_video"+str(currentframe)+".txt", "w+")
                Trans = " ".join(map(str, Trans))
                encoded_file.write(Trans)
                encoded_file.close()
                print("./coded_frames/encoded_video" + str(currentframe) + ".txt" + " created...")

                currentframe += 1
                print("frame"+str(currentframe)+" coded...")
                previous_frame = img1_copy

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        encoded_file = open("./coded_frames/encoded_video.txt", "w+")
        all_coded_frames = numpy.array(all_coded_frames, dtype='int32')
        # all_coded_frames.astype('int16').tofile(encoded_file)
        all_coded_frames = " ".join(map(str, all_coded_frames))
        encoded_file.write(all_coded_frames)
        encoded_file.close()
        huffman=Huffman("./coded_frames/encoded_video.txt")
        huffman.encode()

def make_frame_array(frame_type,Trans,array):
    frame = []
    frame.extend(array)
    frame.append(frame_type)
    frame.append(len(Trans))
    frame.extend(Trans)
    return frame

obj = Encoder("a.avi")
obj.encode()
obj.endCapture()