# -*- coding: utf-8 -*-
"""
Created on May 08 18:00:48 2021

@author: Mayara Everlim Bonani [s6mabona], Ismail Wahdan[s6iswahd]
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA


############# Exercise 1 Functions ##################################

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
def perform_optical_flow(stack,interest_points_first_frame,start,finish):
    d1, d2 ,d3 = interest_points_first_frame.shape
    trajectories = np.zeros((trajectory_length,d1,d2,d3))
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    old_frame = stack[start]
    p0 = interest_points_first_frame
    trajectories[0] = p0
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    for j in range(start+1,finish):
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
        trajectories[j-start] = p0
    
 
    cv2.destroyAllWindows()
              
    for tr_number in range(interest_points_first_frame.shape[0]):
        # first point in track
        x1, y1, = trajectories[0][tr_number][0]
        x1 , y1 = int(x1) , int (y1)
        # last point in track
        x2, y2 = trajectories[14][tr_number][0]
        x2 , y2 = int(x2) , int (y2)

        if stack[start][x1][y1] != stack[finish-1][x2][y2]:
            np.delete(trajectories, tr_number , 1)
    return trajectories



    
##################  Exercise 1 Trajectory Shape Descriptor   ##############################
def trajectoryShapeDescriptor(trajectory, nextPosition):
   
    displacements = []
    normDisplacements = []
    for t in range(trajectory.shape[0]-1):
        displacementVector =  trajectory[t+1] - trajectory[t]
       
        normDisplacements = np.append(normDisplacements, np.linalg.norm(displacementVector))
        displacements = np.append(displacements, displacementVector)
        
    displacementVector = trajectory[trajectory.shape[0]-1] - nextPosition
    displacements = np.append(displacements, displacementVector)
    
    descriptors = displacements / np.sum(normDisplacements)
    return(displacements, descriptors)




##################  Exercise 2 Functions    ######################################################

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
         #print(volume[i].shape)
        
     return(volume)
 
 
   
# 2.2 -  createTubes(volume)
# Function that divide the volume 15x 32× 32 into tubes of size 3x 2× 2 
def createTubes(volume):
    
    tubes = np.zeros((12,5,16,16),dtype = np.uint8)
    i = 0
    for y in range (0,32,16):
        for x in range(0,32,16):
            for t in range(0,15,5):
                
                tubes[i] = volume[t:t+5,x:x+16,y:y+16]
               # print("tubes",i,":")
               # print("t:",t,t+5,"x:",x,x+16,"y:",y,y+16)
               # print(tubes[i])
                i = i+1
    
    return tubes # Return 12 tubes

# Function based on the source code of the HOG function - opencv's examples directory
def hog_for1Cell(img):
    # Derivative in x
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    # Derivative in y
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    # Convert to polar coordinates to find the magnitude of the gradient and the orientation
    mag, ang = cv2.cartToPolar(gx, gy,angleInDegrees=True)
    # The histogram will have 8 bins
    bin_n = 8 # Number of bins
    # For each pixel of the image, convert the ang to a specific bin 
    bin = np.int32(bin_n*ang/(360.))

    hist = np.zeros(bin_n)
    for i in range(int(img.shape[0])):
        for j in range(int(img.shape[1])):
            if(bin[i,j]==8): bin[i,j] = 7 # In case the angle is 360 degrees
            hist[bin[i,j]] = hist[bin[i,j]]+ mag[i,j]

    # The histograms will be normalized after concatenation

    return(hist)

# HOG features
# Return the normalized histogram for all the cells
def hog_allCells(cells):
    allHist = [] # The histogram of all cells
    bin_n = 8 # number of bins
    for ncells in range(cells.shape[0]): # for the total number of cells
        hist = np.zeros(bin_n)
        for t in range(cells.shape[1]): # for all cells.shape[1] frames
            h = hog_for1Cell(cells[ncells][t]) # Calculate the histograms for each frame
            hist =  hist+ h # Sum the histograms for all the cells.shape[1] frames
          
        allHist = np.append(allHist,hist)

    # L2 normalization
    norm = np.linalg.norm(allHist)
    if (norm!=0):
        hogHist =allHist/ np.linalg.norm(allHist)
    else:
        hogHist = allHist
        
    return(allHist,hogHist)



# Function that returns the HOF descriptor
# It receives the cells, the Histogram of Gradients, 
# the displacements between the pixels, and the last frame of the trajectory (tmax)
def hof_allCells(cells,gradHist,displacements,tmax):
    allHist = [] # The HOF histograms for all cells
    bin_n = 9
    for ncells in range(cells.shape[0]): # for the total number of cells
        hist = np.zeros(bin_n)
        hist[0:8] = gradHist[ncells:ncells+8]  
        # For the vector of the displacement of the pixel coordinates
        
        
        threshold = 1.3
        count = 0
        for t in range(0,tmax,2):
            displacementVector  = displacements[t:t+2]
            mag = np.linalg.norm(displacementVector)            
            # if the magnitude is bellow a threshold it increases a bin in the centers
            if (mag < threshold): count = count+1 
            
        hist[8] = count
        
        allHist = np.append(allHist,hist)
        
    # L2 normalization
    norm = np.linalg.norm(allHist)
    
    if (norm!=0):
        hofHist =allHist/ np.linalg.norm(allHist)
    else:
        hofHist = allHist # If all the vectors are 0
    return(hofHist)


# Motion Boundary Histogram in x and y  for 1 cell
def mbh_for1cell(cell):
    
    mhbxHist_1cell = np.zeros(8)
    mhbyHist_1cell = np.zeros(8)
    
    for t in range(cell.shape[0]-1): # For all 5 frames of the cell
        # 1 - Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(cell[t],cell[t+1], None, 0.5,1,16, 3, 5, 1.2, 0)
        optflowx = flow[:,:,0]
        optflowy = flow[:,:,1]
        
        # 2- Compute x-,y-derivatives of optical flow
        
        # For the optical flow in the x direction
        # HOG for 1 cells will Compute x-,y-derivatives of optical flow in x
        mhbxHist_1frame = hog_for1Cell(optflowx)
        mhbxHist_1cell = mhbxHist_1cell + mhbxHist_1frame 
        
        # For the optical flow in the y direction
        # HOG for 1 cells will Compute x-,y-derivatives of optical flow in y
        mhbyHist_1frame = hog_for1Cell(optflowy)
        mhbyHist_1cell = mhbyHist_1cell + mhbyHist_1frame
        
        
    return(mhbxHist_1cell, mhbyHist_1cell)
        
      
        
        

# Motion Boundary Histogram in x and y  for the tubes
def mbh_allCells(cells):
    allHistx = [] # The MBHx histograms for all cells
    allHisty = []
    for ncells in range(cells.shape[0]): # for the total number of cells
        
        histx, histy = mbh_for1cell(cells[ncells])
        allHistx = np.append(allHistx,histx)
        allHisty = np.append(allHisty,histy)
   
    # L2 normalization
    normx = np.linalg.norm(allHistx)
    normy = np.linalg.norm(allHisty)
    if (normx!=0):
        mbhxHist =allHistx/ normx 
    else:
        mbhxHist = allHistx
    
    if (normy!=0):
        mbhyHist =allHisty/ normy 
    else:
        mbhyHist = allHisty
    
   
    return (mbhxHist, mbhyHist)

    
    
    
#################### Main code ##################################### 
        
"""========== MAIN ========== """

video  = cv2.VideoCapture('data/dummy.avi') # It is possible to replace it for the other videos.

##### Exercise 1 #######################################################
# Stack images   
stack = stack_video(video) 

# Calculate Interest points of first frame
interest_points_first_frame = extract_points_from_frame(stack[0])    

# Then use optical flow to calculate a trajectory over these points

trajectory_length=15
#trajectories = perform_optical_flow(stack,interest_points_first_frame,trajectory_length)
trajectories = []
firstInterestPoints = []# Save the interest points for the beginning of each trajectory

i = 1
#for start in range(0,stack.shape[0],15):
for start in range(0,stack.shape[0],15):
    # Extract the interest points
    interest_points_first_frame_of_15 = extract_points_from_frame(stack[start])  # I have to recognize if is the same
    # Save the interest points of the beginning of each trajectory
    firstInterestPoints = np.reshape (np.append(firstInterestPoints,interest_points_first_frame_of_15),(i,interest_points_first_frame_of_15.shape[0],2))
    i = i+1
    # Calculate the trajectory of the interest points for 15 frames
    trajectories_for_15_frames = perform_optical_flow(stack,interest_points_first_frame_of_15,start, min(start+15,stack.shape[0]))
    trajectories.append(trajectories_for_15_frames)

trajectories = np.array(trajectories) # Convert a list to an numpy



##### Exercise 2 +  Exercise 1: Trajectory Shape Descriptor #####################################################################

# 1. Extract all the single trajectories with the respective starting frames

allTubesFeatures = []

# For the all the trajectories of each interest point =  trajectories.shape[0] 
# trid: It indicates the position of the trajectory in the video. Ex: first 15 frames-> trid = 0 
for trid in range(trajectories.shape[0]):
    # For all interest points = trajectories.shape[2]
    #ptointid: It indicates the index of the interest point.
    for ptointid in range(trajectories.shape[2]): 
 
        singleTrajectory = trajectories[trid,:,ptointid,:] 

        # Exercise 1: Calculate the Trajectory Shape Descriptor and the Displacements(u,v)
        if (trid <trajectories.shape[0]-1): 
            displacements, trajectoryDescriptor = trajectoryShapeDescriptor(singleTrajectory,firstInterestPoints[trid+1][ptointid])
        else: # For the lasts frames
             displacements, trajectoryDescriptor = trajectoryShapeDescriptor(singleTrajectory,[0.,0.])
            
    
        # 2. For each point of interest, create the volume: 
        # Around each extracted 15 frame long trajectory, create a volume of 32× 32× 15
        # (where 32× 32 is spatial window, and 15 is the length of trajectory). 
        volume = createVolume(singleTrajectory, stack, startingFrame=trid).astype(np.uint8)                  
        
        # 3-  Divide this volume 32× 32× 15 into tubes of size 2× 2× 3.
        tubes = createTubes(volume)
        
        # 4. Calculate the HOG histograms considering 8 bins
        # 96 dimensional HoG = hogHist
        # hist is the HoG histrogram without the L2 normalization
        hist, hogHist= hog_allCells(tubes) 
        
        # 5. Calculate the HOF histograms considering 9 bins
        # 108 dimensional HoF
        tmax = singleTrajectory.shape[0]
        hofHist = hof_allCells(tubes,hist,displacements,tmax)    
        
        # 6- Calculate the MBHx histograms considering 8 bins
        # 7- Calculate the MBHy histograms considering 8 bins
        # 96 dimesnional MBHx
        # 96 dimensional MBHy
        mbhxHist, mbhyHist = mbh_allCells(tubes) 
        
        
        # 8- Concatenate extracted feature for each tube:
        # 426 dimensional tube Featurevector = 
        # 30 dimensional Trajectory Shape
        # 96 dimensional HoG
        # 108 dimensional HoF
        # 96 dimesnional MBHx
        # 96 dimensional MBHy'
        tubeFeatures = np.concatenate((trajectoryDescriptor, hogHist,hofHist,mbhxHist, mbhyHist), axis=0)


        allTubesFeatures = np.append(allTubesFeatures,tubeFeatures)

# Just test the dimensions for the last trajectory
print("Trajectory Shape Descriptor dimension:",trajectoryDescriptor.shape)
print("Hog Dimension:", hogHist.shape)
print("Hog Dimension", hofHist.shape)
print("MBHx Dimension:",mbhxHist.shape)
print("MBHy Dimension:",mbhyHist.shape)
print("Tube features dimension:", tubeFeatures.shape)

allTubesFeatures = allTubesFeatures.reshape((int(allTubesFeatures.shape[0]/426), 426))
print("Vector with all the trajectories of the video dimension (nsamples,features):", allTubesFeatures.shape)

############# Exercise 3 ##############################################################
#  Calculate the video representation, this involves:  
# Apply PCA on your calculated features, to reduce dimension to 64 from 426
pca = PCA(n_components=64)
allTubesFeaturesReducted =  pca.fit_transform(allTubesFeatures)




