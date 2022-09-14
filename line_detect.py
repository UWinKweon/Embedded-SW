import cv2
import numpy as np

img = cv2.imread('C:/Users/yoosk/Documents/images/hallway_ceiling.jpg')
print(img.shape)    # 이미지 크기 확인용

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
can = cv2.Canny(gray, 50, 200, None, 3)

# 관심영역으로 설정할 좌표값 찾기
'''
x,y,w,h	= cv2.selectROI('img_roi', can, False, False)
print("ROI = (%d, %d), (%d, %d)"%(x, y, x + w, y + h))      # left top, right bottom

if w and h:
    roi = img[y:y+h, x:x+w]
    cv2.imshow('cropped', roi)                  # ROI 지정 영역을 새창으로 표시
    cv2.moveWindow('cropped', 0, 0)             # 새창을 화면 좌측 상단에 이동
'''

# 설정한 좌표값들 원 그려서 확인
'''
circle_img = img.copy()
rectangle = np.array([(30, 0), (180, 420), (260, 420), (120, 0)])
print(rectangle.shape)

# cv2.circle(circle_img, (61, 0), 5, (0, 0, 255), -1)
# cv2.circle(circle_img, (211, 393), 5, (0, 0, 255), -1)
for rec in rectangle:
    cv2.circle(circle_img, rec, 5, (0, 0, 255), -1)
cv2.imshow('circle', circle_img)
'''

# 관심구역 Mask 만들어서 원하는 영역 엣지만 검출
'''
rectangle = np.array([[(30, 0), (180, 420), (260, 420), (120, 0)]])
mask = np.zeros_like(can)               # Canny 엣지 이미지와 같은 크기, 값은 0
cv2.fillPoly(mask, rectangle, 255)      # 설정한 관심 구역 안의 값들 255로 바꿈
cv2.imshow('mask', mask)

masked_img = cv2.bitwise_and(can, mask)                 # shape : (608, 480)
print(masked_img.shape)
cv2.imshow('masked_img', masked_img)

ccan = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)     # shape : (608, 480, 3)
print(ccan.shape)
cv2.imshow('ccan', ccan)
'''

rectangle = np.array([[(30, 0), (180, 420), (260, 420), (120, 0)]])
mask = np.zeros_like(can)                               # Canny 엣지 이미지와 같은 크기, 값은 0
cv2.fillPoly(mask, rectangle, 255)                      # 설정한 관심 구역 안의 값들 255로 바꿈
masked_img = cv2.bitwise_and(can, mask)                 # shape : (608, 480)
ccan = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)     # shape : (608, 480, 3)

line_det = img.copy()
line_arr = cv2.HoughLinesP(can, 1, np.pi/180, 100, minLineLength=10, maxLineGap=10)
print(line_arr.shape, len(line_arr))

line = np.empty((0, 5), int)
print(line, line.shape)

if line_arr is not None:
    line_arr2 = np.empty((len(line_arr), 5), int)
    for i in range(0, len(line_arr)):
        temp = 0
        l = line_arr[i][0]      # 검출된 각각의 직선 좌표값들 가져옴
        line_arr2[i] = np.append(line_arr[i], np.array((np.arctan2(l[1] - l[3], l[0] - l[2]) * 180) / np.pi))
        print("검출된 직선 좌표값 :", line_arr2[i], " 각도 ", np.array((np.arctan2(l[1] - l[3], l[0] - l[2]) * 180) / np.pi))
        if line_arr2[i][1] > line_arr2[i][3]:   # y1 > y2 인 경우 
            temp = line_arr2[i][0], line_arr2[i][1]
            line_arr2[i][0], line_arr2[i][1] = line_arr2[i][2], line_arr2[i][3]
            line_arr2[i][2], line_arr2[i][3] = temp
        if (line_arr2[i][2] < 250 and 100 < abs(line_arr2[i][4]) < 125):
            line = np.append(line, line_arr2[i])
line = line.reshape(int(len(line) / 5), 5)
print(line)

try:
    # line = line[line[:, 0].argsort()[-1]]
    degree = line[0][4]
    print(degree)
    cv2.line(ccan, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    mimg = cv2.addWeighted(img, 1, ccan, 0.9, 0)
except:
    degree = 0

cv2.imshow('ccan_result', ccan)
cv2.imshow('mimg', mimg)


for line_d in line_arr:
    x1, y1, x2, y2 = line_d[0]
    cv2.line(line_det, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow('IMG', img)
cv2.imshow('GRAY', gray)
cv2.imshow('CANNY', can)
cv2.imshow('Line Draw', line_det)
cv2.waitKey(0)
cv2.destroyAllWindows()