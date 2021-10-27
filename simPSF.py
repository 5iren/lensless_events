import cv2
import argparse
import numpy as np
from utils import framePreprocess, fftConvolve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video_name",   help="name of the video", required=True)
    parser.add_argument("-r", "--frame_rate",   help="name of the video", default=30, type=int)

    #Get arguments
    args = parser.parse_args()
    video_name = args.video_name
    fps = args.frame_rate
    
    #Camera settings (DAVIS camera)
    davis_width = 346
    davis_height = 260
    davis_ratio = davis_height / davis_width

    #Directory settings
    video_dir = "data/original_videos/"
    results_dir = "data/convolved_videos/"
    cropped_dir = "data/cropped_videos/"
    video_path = video_dir + video_name
    single_name = video_name.split('.')
    result_name = results_dir + single_name[0] + '.avi'
    cropped_name = cropped_dir + single_name[0] + '.avi'
    psf_path = 'data/psf/psf_16bit_baffle.tif'
    # psf_path = 'data/psf/pinholePSF.png'
    #psf_path = 'data/psf/delta2.png'

    
    #Print arguments
    print("---------------------------------------")
    print("Video name:        ", video_name)
    print("Video directory:   ", video_path)
    print("Results directory: ", result_name)
    print("Frame size:         {}x{}".format(davis_width, davis_height))
    print("FPS:               ", fps)
    print("---------------------------------------")

    #Create PSF 
    # psf = cv2.imread(psf_path, 0)
    # psf = cv2.resize(psf, (davis_width, davis_height)) #resize to sensor shape
    # psf = psf / psf.sum() #Normalize

    psf = np.load("data/psf/psf_3.npy")
    psf /= psf.max()
    psf *= 255
    psf = psf.astype('uint8')
    psf = cv2.cvtColor(psf, cv2.COLOR_RGB2GRAY)
    psf = cv2.resize(psf, (davis_width, davis_height))
    psf = psf / psf.sum() #Normalize

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
            #print(lensless.max())
            lensless = np.uint8(lensless*255)
            #print(lensless.max())
            #cv2.imwrite("lensless.png", lensless)

            #Create video with write
            conv_out.write(lensless)

        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    crop_out.release()
    conv_out.release()

    print("[INFO] Done!")




