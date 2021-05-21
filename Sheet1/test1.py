# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:00:48 2021

@author: Ricardo Landim
"""

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
    #
    count = 0
    #
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
     #  Save the frames instead of showing them 
        cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file 
        count = count+1
    #
        #cv2.imshow('frame',img)
        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
        #    break
        # Now update the previous frame and previous points
        old_frame = frame.copy()
        p0 = good_new.reshape(-1,1,2)
        trajectories[j] = p0
    return trajectories
 
################## Exercise 2 Functions       ######

# 2.1 -  createVolume(singleTrajectory, stack, startingFrame)
# Function that creates a volume of 32× 32× 15:
# - 32× 32 is spatial window
# - 15 is the length of trajectory
# - stack = vector with all the frames
# - startingFrame = frame of the initial point of the trajectory
def createVolume(singleTrajectory, stack, startingFrame):
     # frames = all the 15 frames of the trajectory 
     # From the starting frame to starting frame + 15
     frames = stack[startingFrame:startingFrame+15] 
     # Volume of size 15x32x32
     volume = np.zeros((15,32,32))
     # Size of each frame
     imgXsize, imgYsize = stack[0].shape 
     
     for i in range(15):# For each one of the 15 frames
         # For each point of interest with coordinates x and y
         x = int(singleTrajectory[i,0,0]) # I am not sure about x or y
         y = int(singleTrajectory[i,0,1])
         # For the limits of the volume, centering x and y.
         limit_xNeg= x-16
         limit_xPos= x+16
         limit_yNeg = y-16
         limit_yPos = y+16
         # Test if the limits are within the limits of the image
         # If the limits are out the image, compensate the limits of the volume
         # to always have 32x32 pixels
         if (limit_xNeg< 0): 
             dif = -1* limit_xNeg
             limit_xNeg =0
             limit_xPos = limit_xPos + dif # compensate to keep the number of cells constant 32x32
         if (limit_yNeg< 0): 
             dif = -1* limit_yNeg
             limit_yNeg =0
             limit_yPos = limit_yPos + dif
         if (limit_xPos > imgXsize-1):
             dif = limit_xPos - (imgXsize-1)
             limit_xPos = imgXsize-1
             limit_xNeg = limit_xNeg-dif
         if (limit_yPos > imgYsize-1): 
             dif = limit_yPos - (imgYsize-1)
             limit_yPos = imgYsize-1
             limit_yNeg = limit_yNeg-dif
         
         #print (limit_xNeg,limit_xPos, limit_xNeg-limit_xPos  , limit_yNeg,limit_yPos,limit_yNeg-limit_yPos)
         # Create the volume for the frame i
         volume[i] = frames[i][ limit_xNeg:limit_xPos , limit_yNeg:limit_yPos ]
         print(volume[i].shape)
        
     return(volume)
 
 
   
# 2.2 -  createTubes(volume)
# Function that divide the volume 15x 32× 32 into tubes of size 3x 2× 2 
def createTubes(volume):
    tubes = np.zeros((1280,3,2,2),dtype = np.uint8)
    i = 0
    for y in range (0,32,2):
        for x in range(0,32,2):
            for t in range(0,15,3):
                
                tubes[i] = volume[t:t+3,x:x+2,y:y+2]
                print("tubes",i,":")
                print("t:",t,t+3,"x:",x,x+2,"y:",y,y+2)
                print(tubes[i])
                i = i+1
    
    return tubes # Return 1280 tubes
    
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


##### Exercise 2 #####################################################################
# Around each extracted 15 frame long trajectory, create a volume of 32× 32× 15
# (where 32× 32 is spatial window, and 15 is the length of trajectory). 
# 1. For each point of interest, create the volume: 
#    number of points of interest =  trajectories.shape[1]

#TODO - Extract all the single trajectories with the respective starting frames
#TODO - Process for every single trajectory all the descriptors
# Coordinates x and y of a single trajectory of 15 frames
singleTrajectory = trajectories[:,0,:,:] # for the point of interest 0
print(singleTrajectory.shape)

# Create a volume of size 15x32x32
volume = createVolume(singleTrajectory, stack, startingFrame=0).astype(np.uint8)                  

# Divide this volume 32× 32× 15 into tubes of size 2× 2× 3.
tubes = createTubes(volume) # I dont know if it is necessary

# Test for one image
img = volume[0]
cell_size = (2,2)
block_size = (2, 2)  # h x w in cells
nbins = 8
# winSize is the size of the image cropped to an multiple of the cell size
# cell_size is the size of the cells of the img patch over which to calculate the histograms
# block_size is the number of cells which fit in the patch
hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)


h = hog.compute(img)


# HoG with 8 bins -  8 possible orientations

# hog = cv2.HOGDescriptor()
# h = hog.compute()

# HoF with 9 bins 
# MBHx with 8 bins 
# MBHy with 8 bins

#Concatenate extracted feature for each tube
# 96 dimensional HoG
# 108 dimensional HoF
# 96 dimesnional MBHx
# 96 dimensional MBHy




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