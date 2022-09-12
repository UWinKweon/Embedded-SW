import cv2
import numpy as np

img = cv2.imread('C:/Users/yoosk/Documents/images/hallway_ceiling.jpg')     # 480 x 608
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
can = cv2.Canny(gray, 50, 200, None, 3)
line_arr = cv2.HoughLinesP()

if img is not None:
    cv2.imshow('IMG', img)
    cv2.imshow('GRAY', gray)
    cv2.imshow('CANNY', can)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('No image file.')