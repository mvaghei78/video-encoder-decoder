
# Importing all necessary libraries
import cv2
import os

class GrayScale:
    def __init__(self, videopath):
        # Read the video from specified path
        self.video = cv2.VideoCapture(videopath)
        ret, frame = self.video.read()

        self.height, self.width, self.nchannels = frame.shape
        fps = 25
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('grayscale_output.avi', self.fourcc, fps, (self.width, self.height), 0)
        try:
            # creating a folder named data
            if not os.path.exists('data'):
                os.makedirs('data')
        # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')

    def extractImages(self):
        currentframe = 0
        while (True):
            # reading from frame
            ret, frame = self.video.read()

            if ret:
                # if video is still left continue creating images
                name = './data/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #save gray fram in output video file
                self.out.write(gray)

                # Display the resulting frame
                cv2.imshow('Live',gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break

    def endCapture(self):
        # Release all space and windows once done
        self.out.release()
        self.video.release()
        cv2.destroyAllWindows()

obj = GrayScale("sample.mp4")
obj.extractImages()
obj.endCapture()