import cv2
import numpy as np



def extract_points_from_frame(frame):
    # Detector parameters
    blockSize = 3
    apertureSize = 5
    k = 0.04
    # Detecting corners
    dst = cv2.cornerHarris(frame, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    result = np.zeros((dst_norm.shape))
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > 128:
                result[i][j] = 255
            else:
                result[i][j]=0
    cv2.namedWindow('Corners detected')
    cv2.imshow('Corners detected', dst_norm_scaled)
    return result



def stack_video(video):
    # get first frame to know measurements
    ret, frame1 = video.read()
    
    #convert to grayscale     
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    stack = np.array([frame1])
    
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            stack = np.vstack((stack,[frame]))
        else: 
              break
        
    return stack


# needs debugging 
def perform_optical_flow(stack_of_interest_points):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    old_frame = stack_of_interest_points[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    for frame in stack:
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

#main
video  = cv2.VideoCapture('data/dummy.avi')

# EX1
# Stack images   
stack = stack_video(video)

interest_points_throught_time = np.zeros((stack.shape))
for i,frame in enumerate(stack):
    extracted_points= extract_points_from_frame(frame)
    interest_points_throught_time[i] = extracted_points
    
    #sanity check, 4 points should be found around corners 
    print(extracted_points[extracted_points>0].size/4 == 4)
    
    

perform_optical_flow(stack)
"""
#Show video
i=0
video  = cv2.VideoCapture('data/dummy.avi')
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        i+=1
        extract_points_from_frame(frame)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break
    
print("#Frames check", i)  
video.release()
    
cv2.destroyAllWindows()
"""