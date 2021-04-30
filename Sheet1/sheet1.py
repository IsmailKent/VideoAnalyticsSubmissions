import cv2
import numpy as np

video  = cv2.VideoCapture('data/dummy.avi')

if (video.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(video.isOpened()):
  ret, frame = video.read()
  if ret == True:

    cv2.imshow('Frame',frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else: 
    break

video.release()

cv2.destroyAllWindows()