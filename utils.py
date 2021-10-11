import cv2
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
    