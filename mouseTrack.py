import cv2 
import numpy as np    

def main():    
    # define a video capture object 
    vid = cv2.VideoCapture(0) 
    
    #initialize frame and xy resolution varbiables
    ret, frame = vid.read()
    yRes, xRes, depth = frame.shape
    print(frame.shape)

    # initialize the array to hold the two sequential frames
    # Reduce the frame size to 1/10th the original and store the resulting dx dy resolutions
    fDisp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f1 = cv2.resize(fDisp, (int(xRes/10), int(yRes/10)), interpolation = cv2.INTER_AREA)
    f2 = f1.copy()
    f = [f1, f2]
    dyRes, dxRes = f1.shape
    
    # Initialize the array to hold the cropped frames for delta comparing
    fCrop = [f1[8:dyRes-16, 8:dxRes-16], f2[8:dyRes-16, 8:dxRes-16]]
    print(fCrop[0].shape)


    # iniitlize the index (this will flip between 1 and 0 each frame) 
    fidx = 0

    # initialize the scores matrix; 16x16 means it will search from -8 to +7 pixels (likely too small) 
    scores = np.zeros((16,16))

    # Initialize the debug/display image
    fDisp = np.zeros_like(cv2.resize(fDisp, ((dxRes+2)*16, (dyRes+2)*16)))

    # Start the frame tracking loop
    while(True): 
        # Reset the debug/display image
        fDisp = np.zeros_like(fDisp)
        
        # Capture the video frame by frame 
        ret, frame = vid.read()

        # Convert to grayscale (would be better if camera could give us a grayscale frame) 
        fGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale the original camera image 
        f[fidx==1] = cv2.resize(fGray, (dxRes, dyRes), interpolation = cv2.INTER_AREA)

        # Get the cropped image from the center of the original frame in the other index
        fCrop[0] = f[fidx!=1][(8+0):(dyRes-8)+0, (8+0):(dxRes-8)+0]

        # iterate over dx, dy to compute each score
        for dx in range(-8,8):
            for dy in range(-8,8):
                # Get the cropped image at this dx,dy offset from the new frame
                fCrop[1] = f[fidx==1][(8+dy):(dyRes-8)+dy, (8+dx):(dxRes-8)+dx]
                #print(fCrop[1].shape)

                # Compute the score
                fDiff = cv2.absdiff(fCrop[0], fCrop[1])
                dIdx = (8+dy, 8+dx)
                scores[dIdx] = np.sum(fDiff)

                # Copy the diff image into the display/debug output image
                fDisp[(dIdx[0]*dyRes):(dIdx[0]*dyRes)+dyRes-16, (dIdx[1]*dxRes):(dIdx[1]*dxRes)+dxRes-16] = fDiff
                #fDisp[(dIdx[0]*(dyRes+2)):(dIdx[0]*(dyRes+2))+dyRes, (dIdx[1]*(dxRes+2)):(dIdx[1]*(dxRes+2))+dxRes] = cv2.resize(fDiff, (dxRes, dyRes), interpolation = cv2.INTER_AREA)

        # find the index of the minimum value
        minIdx = np.unravel_index(np.argmin(scores, axis=None), scores.shape)

        # Draw a red box around the frame with the minimum value
        fDisp = cv2.rectangle(fDisp, (minIdx[1]*dxRes, minIdx[0]*dyRes), ((minIdx[1]*dxRes)+dxRes, (minIdx[0]*dyRes)+dyRes), (255, 0, 0), 2)
        cv2.imshow('frame', fDisp) 
        
        # Undo the red box by drawing a block box over it in the same place
        fDisp = cv2.rectangle(fDisp, (minIdx[1]*dxRes, minIdx[0]*dyRes), ((minIdx[1]*dxRes)+dxRes, (minIdx[0]*dyRes)+dyRes), (0, 0, 0), 2)

                
        # use the 'c' button to capture the image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('frame.png', frame)
            print('Image Captured')
        
        # the 'q' button is set as the quitting button 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

        # Flip the index to the otheer value
        fidx = (fidx == 0)

    # After the loop release the cap object 
    vid.release() 
    
    
    # Destroy all the windows 
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()