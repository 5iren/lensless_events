import cv2
import os
import argparse
import numpy as np

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

def framePreprocess(frame, davis_height, davis_width, davis_ratio):
    #Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Get shape
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    frame_ratio = frame_height / frame_width

    #If frame_ratio > davis_ratio: resize width
    if frame_ratio > davis_ratio:
        #Get new frame and width from ratio
        new_height = int(davis_width * frame_ratio)
        new_width = davis_width

        #Resize
        frame = cv2.resize(frame, (new_width, new_height))

        #Center crop
        height_difference = new_height - davis_height
        cropped_frame = frame[int(height_difference/2):int(height_difference/2+davis_height),:]

    #If frame_ratio < davis_ratio: resize height
    elif frame_ratio < davis_ratio:
        #Get new frame and width from ratio
        new_height = davis_height
        new_width = int(davis_height / frame_ratio)
        
        #Resize
        frame = cv2.resize(frame, (new_width, new_height))

        #Center crop
        width_difference = new_width - davis_width
        cropped_frame = frame[:, int(width_difference/2):int(width_difference/2+davis_width)]

    assert cropped_frame.shape[0] == davis_height, 'Height is not '+str(davis_height)+". Frame height: " +str(frame.shape[0])
    assert cropped_frame.shape[1] == davis_width, 'Width is not '+str(davis_width)+". Frame width: " +str(frame.shape[1])

    return cropped_frame

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
    frame_fft = np.fft.ifft2(frame_fft)
    frame_fft = np.fft.ifftshift(frame_fft)

    return frame_fft
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video_name",   help="name of the video", required=True)

    #Get arguments
    args = parser.parse_args()
    video_name = args.video_name
    
    #Camera settings (DAVIS camera)
    davis_width = 346
    davis_height = 260
    davis_ratio = davis_height / davis_width
    fps = 40

    #Directory settings
    video_dir = "data/original_videos/"
    results_dir = "data/convolved_videos/"
    cropped_dir = "data/cropped_videos/"
    video_path = video_dir + video_name
    single_name = video_name.split('.')
    result_name = results_dir + single_name[0] + '.avi'
    cropped_name = cropped_dir + single_name[0] + '.avi'

    
    #Print arguments
    print("---------------------------------------")
    print("Video name:        ", video_name)
    print("Video directory:   ", video_path)
    print("Results directory: ", result_name)
    print("Frame size:         {}x{}".format(davis_width, davis_height))
    print("FPS:               ", fps)
    print("---------------------------------------")

    #Crop video
    # print("[INFO] Cropping Video...")
    # out_w = davis_width * 3 #widht of output rectangle
    # out_h = davis_height * 3 #height of output rectangle
    # x = 0 #top left corner
    # y = 0
    # os.system('ffmpeg -i '+video_path+' -filter:v "crop='+str(out_w)+':'+str(out_h)+':'+str(x)+':'+str(y)+'" '+cropped_name)

    #Create PSF (2D Gaussian distribution)
    #psf = makeGaussian(frame_size, radius)
    psf = np.zeros((davis_height, davis_width))
    psf[davis_height//2,davis_width//2] = 1

    #Create video object
    cap = cv2.VideoCapture(video_path)

    #Create video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    crop_out = cv2.VideoWriter(cropped_name, fourcc, fps, (davis_width, davis_height), 0)
    conv_out = cv2.VideoWriter(result_name, fourcc, fps, (davis_width, davis_height), 0)


    #Check if video opened succesfully
    if (cap.isOpened()==False):
        print("[ERROR] Cannot open video")

    #Loop through video frames, resize to davis dimensions, and convolve
    print("[INFO] Performing convolution and writing video...")
    while(cap.isOpened()):
        #Read frame by frame
        ret, frame = cap.read()
        
        #If read succesful
        if ret == True:
            # Preprocess frame
            frame = framePreprocess(frame, davis_height, davis_width, davis_ratio)  

            # Save cropped video
            crop_out.write(frame)

            #Compute convolution
            lensless = fftConvolve(frame/255., psf)
            lensless = np.uint8(lensless*255)

            #Create video with write
            conv_out.write(lensless)

        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    crop_out.release()
    conv_out.release()

    print("[INFO] Done!")




