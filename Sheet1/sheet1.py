import cv2
import numpy as np


# Returns  2D list of corner points in image
def extract_points_from_frame(frame):
    # Detector parameters
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 5,
                           blockSize = 5 )
    result = cv2.goodFeaturesToTrack(frame, mask = None, **feature_params)
    
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



#  collect points and return them in list of trajectories 
def perform_optical_flow(stack,interest_points_first_frame,trajectory_length):
    d1, d2 ,d3 = interest_points_first_frame.shape
    trajectories = np.zeros((trajectory_length,d1,d2,d3))
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    old_frame = stack[0]
    p0 = interest_points_first_frame
    trajectories[0] = p0
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    for j in range(1,trajectory_length):
        frame = stack[j]
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
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
        old_frame = frame.copy()
        p0 = good_new.reshape(-1,1,2)
        trajectories[j] = p0
    return trajectories
    
    
    
    
    
"""========== MAIN ========== """


video  = cv2.VideoCapture('data/dummy.avi')

# EX1
# Stack images   
stack = stack_video(video)

# Calculate Interest points of first frame
interest_points_first_frame = extract_points_from_frame(stack[0])    

# Then use optical flow to calculate a trajectory over these points

trajectory_length=15
trajectories = perform_optical_flow(stack,interest_points_first_frame,trajectory_length)

# I think the 30 dimensional descriptor of trajectory is for 15 frames * (x,y) for each frame
# so we a need a tensor of shape (#interest_points, 15*2)
trajectories_descriptor = trajectories.reshape(trajectory_length,interest_points_first_frame.shape[0],-1)
trajectories_descriptor = np.swapaxes(trajectories_descriptor,0,1).reshape(interest_points_first_frame.shape[0], -1)
print(trajectories_descriptor.shape == (4,30))


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