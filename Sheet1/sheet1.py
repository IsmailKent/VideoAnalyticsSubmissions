import cv2
import numpy as np



def extract_points_from_frame(frame):
    # Detector parameters
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blockSize = 3
    apertureSize = 5
    k = 0.04
    # Detecting corners
    dst = cv2.cornerHarris(frame, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    result = np.copy(dst_norm_scaled)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > 255:
                cv2.circle(dst_norm_scaled, (j,i), 5, (0), 2)
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

#main
video  = cv2.VideoCapture('data/dummy.avi')

# EX1
# Stack images   
stack = stack_video(video)



video  = cv2.VideoCapture('data/dummy.avi')
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        i+=1
        interest_points = extract_points_from_frame(frame)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break
video.release()
cv2.destroyAllWindows()
  
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