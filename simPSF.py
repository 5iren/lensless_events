import cv2
import argparse
import numpy as np
from utils import framePreprocess, fftConvolve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video_name",   help="name of the video",           required=True)
    parser.add_argument("-r", "--frame_rate",   help="framerate of lensless video", default=30,     type=int)
    parser.add_argument("-wt", "--width",        help="width of sensor",             default=346,    type=int)
    parser.add_argument("-ht", "--height",       help="height of sensor",            default=260,    type=int)

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
    result_name = results_dir + single_name[0] + '.avi'
    cropped_name = cropped_dir + single_name[0] + '.avi'
    psf_path = 'data/videos_dataset/psf/psf_16bit_baffle.tif'
    #psf_path = 'data/videos_dataset/psf/pinholePSF.png'
    #psf_path = 'data/videos_dataset/psf/delta2.png'

    
    #Print arguments
    print("---------------------------------------")
    print("Video name:        ", video_name)
    print("Video directory:   ", video_path)
    print("Results directory: ", result_name)
    print("Frame size:         {}x{}".format(sensor_width, sensor_height))
    print("FPS:               ", fps)
    print("---------------------------------------")

    #Create PSF 
    psf = cv2.imread(psf_path, 0)
    psf = cv2.resize(psf, (sensor_width, sensor_height)) #resize to sensor shape
    psf = psf / psf.sum() #Normalize

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
            frame = framePreprocess(frame, sensor_height, sensor_width) 

            # Save cropped video
            crop_out.write(frame)

            #Compute convolution
            lensless = fftConvolve(frame/255., psf)
            lensless = np.uint8(lensless*255)
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




