import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(1)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Define the codec and create VideoWriter Object
# fourcc = cv2.VideoWriter_fourcc(*'H264')
# out = cv2.VideoWriter('output.avi',fourcc, 15.0, (640,480))

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Bluring filter
    blur = cv2.blur(frame,(5,5))
    blur_gray = cv2.blur(gray,(5,5))
    # Canny edge detection
    canny = cv2.Canny(frame,100,200)
    canny_gray = cv2.Canny(gray,100,200)
    canny_blur = cv2.Canny(blur,100,200)
    if ret == True:

        # write to file
        # out.write(frame)

        # Display the resulting frame
        cv2.imshow('Original',frame)
        cv2.imshow('Gray Original', gray)
        cv2.imshow('Blur',blur)
        cv2.imshow('Blur Gray', blur_gray)
        cv2.imshow('Canny', canny)
        cv2.imshow('Canny Gray', canny_gray)
        cv2.imshow('Canny Blur', canny_blur)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

    # Break the loop
    else: 
        break
 
# When everything done, release the video capture object
cap.release()
# out.release()
 
# Closes all the frames
cv2.destroyAllWindows()
