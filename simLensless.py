#Takes standard video (-i) and convolves with PSF to generate lensless video.
#Both lensless and standard video are returned with same width (-wt) and height (-ht)


import cv2
import argparse
import numpy as np
from src.psf_utils import match_dim, load_psf, fftConvolve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video_name",   help="name of the video",           required=True)
    parser.add_argument("-r", "--frame_rate",   help="framerate of lensless video", default=30,     type=int)
    parser.add_argument("-wt", "--width",       help="width of sensor",             default=346,    type=int)
    parser.add_argument("-ht", "--height",      help="height of sensor",            default=260,    type=int)

    #Get arguments
    args = parser.parse_args()
    video_name = args.video_name
    fps = args.frame_rate
    sensor_width = args.width
    sensor_height = args.height
    sensor_ratio = sensor_height / sensor_width
    

    #Directory settings
    video_dir = "data/videos_dataset/original_videos/"
    results_dir = "data/videos_dataset/convolved_videos/"
    cropped_dir = "data/videos_dataset/cropped_videos/"
    video_path = video_dir + video_name
    single_name = video_name.split('.')
    result_name = results_dir + "lensless_" + single_name[0] + '.avi'
    cropped_name = cropped_dir + single_name[0] + '.avi'
    psf_path = 'data/videos_dataset/psf/psf_16bit_baffle.tif'
    #psf_path = 'data/videos_dataset/psf/pinholePSF.png'


    
    #Print arguments
    print("---------------------------------------")
    print("Video name:        ", video_name)
    print("Video directory:   ", video_path)
    print("Results directory: ", result_name)
    print("Frame size:         {}x{}".format(sensor_width, sensor_height))
    print("FPS:               ", fps)
    print("---------------------------------------")

    #Load and resize PSF 
    psf = load_psf(psf_path)
    psf = cv2.resize(psf, (sensor_width, sensor_height))
    psf /= psf.sum() #Normalize
    psf = match_dim(psf, (sensor_height*2, sensor_width*2)) #Adds padding to avoid 

    #Create video object
    cap = cv2.VideoCapture(video_path)

    #Create video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    crop_out = cv2.VideoWriter(cropped_name, fourcc, fps, (sensor_width, sensor_height), 0)
    conv_out = cv2.VideoWriter(result_name, fourcc, fps, (sensor_width, sensor_height), 0)

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
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame.astype('float64') / 255.0
            cropped_frame = cv2.resize(frame, (sensor_width, sensor_height))
            frame = match_dim(cropped_frame, (sensor_height*2, sensor_width*2))

            # Save cropped video
            crop_out.write(np.uint8(cropped_frame*255))

            #Compute convolution
            lensless = fftConvolve(frame, psf)
            lensless = match_dim(lensless, (sensor_height, sensor_width))

            #Create video with write
            conv_out.write(np.uint8(lensless*255))

        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    crop_out.release()
    conv_out.release()

    print("[INFO] Done!")




