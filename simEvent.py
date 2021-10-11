import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    full_gaussian = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    gaussian = full_gaussian / full_gaussian.sum()
    return  gaussian


def fftConvolve(frame, psf):
    """ Convolve frame and PSF to simulate lensless image

    frame and psf must be of the same square shape
    """
    #Get fourier transform of PSF
    psf_fft = np.fft.fft2(psf)

    #Get fourier trasnform of frame
    frame_fft = np.fft.fft2(frame)

    #Product
    frame_fft = frame_fft * psf_fft

    #Inverse and shift
    frame_fft = np.fft.irfft2(frame_fft)
    frame_fft = np.fft.ifftshift(frame_fft)

    return frame_fft
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video_name",   help="name of the video", required=True)

    #Get arguments
    args = parser.parse_args()
    video_name = args.video_name
    
    #Camera settings
    davis_width = 346
    davis_height = 260
    fps = 30

    #Directory settings
    video_dir = "data/videos/"
    results_dir = "results/convolved_videos/"
    cropped_dir = "results/cropped_videos/"
    video_path = video_dir + video_name
    single_name = video_name.split('.')
    result_name = results_dir + single_name[0] + '.avi'
    cropped_name = cropped_dir + single_name[0] + '.mp4'

    
    #Print arguments
    print("---------------------------------------")
    print("Video name:        ", video_name)
    print("Video directory:   ", video_path)
    print("Results directory: ", result_name)
    print("Frame size: {}x{}".format(davis_width, davis_height))
    print("FPS:               ", fps)
    print("---------------------------------------")

    #Crop video
    print("[INFO] Cropping Video...")
    out_w = davis_width * 3 #widht of output rectangle
    out_h = davis_height * 3 #height of output rectangle
    x = 0 #top left corner
    y = 0
    os.system('ffmpeg -i '+video_path+' -filter:v "crop='+str(out_w)+':'+str(out_h)+':'+str(x)+':'+str(y)+'" '+cropped_name)

    #Create PSF (2D Gaussian distribution)
    #psf = makeGaussian(frame_size, radius)
    psf = np.zeros((davis_height, davis_width))
    psf[davis_height//2][davis_width//2] = 1


    #Create video object
    cap = cv2.VideoCapture(cropped_name)

    #Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(result_name, fourcc, fps, (davis_width, davis_height), 0)


    #Check if video opened succesfully
    if (cap.isOpened()==False):
        print("[ERROR] Cannot open video")

    #Read until video is completed
    print("[INFO] Performing convolution and writing video...")
    while(cap.isOpened()):
        #Read frame by frame
        ret, frame = cap.read()
        
        #If read succesful
        if ret == True:
            #Convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (davis_width, davis_height))
            frame = frame.astype(np.float32) / 255

            #Compute convolution
            lensless = fftConvolve(frame, psf)
            lensless = np.uint8(lensless*255)

            #Create video with write
            out.write(lensless)

        else:
            break


    cv2.destroyAllWindows()
    cap.release()
    out.release()

    print("[INFO] Done!")




