import cv2 
import numpy as np    

def find_white_balls(frame):
    #range of white color in RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # define the upper and lower bounds of the white color in RGB
    # used matlab to get the values
    lower_white = np.array([125,125,150])
    upper_white = np.array([255,255,255])
    # create a mask for the white color,  
    # blacks out everything but the white color
    mask_ball = cv2.inRange(frame, lower_white, upper_white)
    # apply a Gaussian blur to the mask
    blurred = cv2.GaussianBlur(mask_ball, (5, 5), 0)
    # apply a bitwise AND to the mask and the frame
    # this masks the frame to only show the white color in original frame
    output_ball = cv2.bitwise_and(frame, frame, mask = mask_ball)
    # find contours in the mask
    contoursball, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area
    minA = 500
    maxA = 20000
    filteredContours = [cnt for cnt in contoursball if minA < cv2.contourArea(cnt) < maxA]
    # create a blank image to draw the contours on
    # filteredMask = np.zeros_like(mask_ball)  
    cv2.drawContours(frame, filteredContours, -1, (0, 255, 0), thickness= 5)
    
    return frame
    
def blob_detection(frame):
    # trying blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 5
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(frame)
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    return frame    
    
def main():

 
    '''
    # read image
    frame = cv2.imread('frame.png')
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = find_white_balls(frame)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    # frame = blob_detection(gray_frame)
    '''
    
    # define a video capture object 
    vid = cv2.VideoCapture(1) 

    while(True): 
        
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
        frame = find_white_balls(frame)
        # Display the resulting frame 
        cv2.imshow('frame', frame) 
        
        # use the 'c' button to capture the image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('frame.png', frame)
            print('Image Captured')
        
        # the 'q' button is set as the quitting button 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()