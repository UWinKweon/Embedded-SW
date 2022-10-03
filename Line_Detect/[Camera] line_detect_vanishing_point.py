from collections import deque
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob     # 운영체제와의 상호작용, 파일들의 리스트를 뽑을때 사용
from moviepy.editor import VideoFileClip    # 비디오 처리
import math

# Threshold by which lines will be rejected wrt the horizontal
REJECT_DEGREE_TH = 4.0      # degree 4~86


def select_white_yellow(image):     # while/yellow 부분만 남김
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)      # RGB to HLS
    # white color mask
    lower = np.uint8([0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([10,   0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)      # mask 영역의 두 이미지를 합침
    return cv2.bitwise_and(image, image, mask=mask)     # mask 영역에서 서로 공통으로 겹치는 부분 출력


def filter_region(image, vertices):     # line detect 결과를 흰색으로 표시/select_region()에서 호출
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:        # garyscale인 경우/len(mask.shape)은 차원 수를 의미
        cv2.fillPoly(mask, vertices, 255)       # 관심구역을 흰색으로 칠함
    # else:     # channel이 있는 경우(RGB)
    #     # in case, the input image has a channel dimension
    #     cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
    return cv2.bitwise_and(image, mask) # bitwise_and() 연산으로 line detect 부분만 추출


def select_region_line(image):       # line detect 관심구역 설정
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]        # height, width
    bottom_left = [cols*0.0, rows*0.90]
    top_left = [cols*0.45, rows*0.6]
    bottom_right = [cols*1.0, rows*0.90]
    top_right = [cols*0.55, rows*0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array(        # 4x2 행렬로 만들어줌
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


def FilterLines(Lines):
    FinalLines = []
    cntLines = []

    for Line in Lines:
        [[x1, y1, x2, y2]] = Line       # 시작점과 끝점

        # Calculating equation of the line: y = mx + c
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = 100000000
        c = y2 - m*x2
        # theta will contain values between -90 ~ +90.
        theta = math.degrees(math.atan(m))      # radian to degree

        # Rejecting lines of slope near to 0 degree or 90 degree and storing others
        if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):       # degree가 4~86인 경우
            l = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)    # length of the line
            FinalLines.append([x1, y1, x2, y2, m, c, l])
        elif abs(theta) < REJECT_DEGREE_TH:     # degree가 4보다 작은 경우
            l = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            cntLines.append([x1, y1, x2, y2, m, c, l])

    # Removing extra lines
    # (we might get many lines, so we are going to take only longest 15 lines
    # for further computation because more than this number of lines will only
    # contribute towards slowing down of our algo.)
    if len(FinalLines) > 15:        # detected line이 15개 이상이면
        # x[-1] : FinalLines의 제일 마지막 값 = l, reverse = True : 내림차순 -> 길이로 내림차순
        FinalLines = sorted(FinalLines, key=lambda x: x[-1], reverse=True)      # 길이를 기준으로 내림차순 정렬하여
        FinalLines = FinalLines[:15]        # 길이가 긴 15개만 배열에 저장
    if len(cntLines) > 15:
        cntLines = sorted(cntLines, key=lambda x: x[-1], reverse=True)
        cntLines = cntLines[0]      # 길이가 가장 긴 가로선만 추출
    return FinalLines, cntLines


def GetVanishingPoint(Lines):       # Vanishing point 구하기
    # We will apply RANSAC inspired algorithm for this. We will take combination
    # of 2 lines one by one, find their intersection point, and calculate the
    # total error(loss) of that point. Error of the point means root of sum of
    # squares of distance of that point from each line.
    VanishingPoint = None
    MinError = 100000000000

    for i in range(len(Lines)):
        for j in range(i+1, len(Lines)):
            m1, c1 = Lines[i][4], Lines[i][5]
            m2, c2 = Lines[j][4], Lines[j][5]

            if m1 != m2:
                x0 = (c1 - c2) / (m2 - m1)
                y0 = m1 * x0 + c1

                err = 0
                for k in range(len(Lines)):
                    m, c = Lines[k][4], Lines[k][5]
                    m_ = (-1 / m)
                    c_ = y0 - m_ * x0

                    x_ = (c - c_) / (m_ - m)
                    y_ = m_ * x_ + c_

                    l = math.sqrt((y_ - y0)**2 + (x_ - x0)**2)

                    err += l**2

                err = math.sqrt(err)

                if MinError > err:
                    MinError = err
                    VanishingPoint = [x0, y0]

    return VanishingPoint


def average_slope_intercept(lines):     # 가중치(길이)가 높은 left_line/right line을 1개씩 추출/lane_lines()에서 호출
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        # if line is not None:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights,  left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):     # 각 line의 양 끝점 좌표값 반환/lane_lines()에서 호출
    if line is None:
        return None

    slope, intercept = line

    if slope != 0:
    # make sure everything is integer as cv2.line requires it
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))
    else:
        return None

def lane_lines(image, lines):       # line의 양 끝점 ((x1,y1),(x2,y2)) 좌표값 구하기
    left_lane, right_lane = average_slope_intercept(lines)      # m,c값 반환

    y1 = image.shape[0]  # bottom of the image = height
    y2 = y1*0.6         # slightly lower than the middle

    # 각 line의 양 끝점 좌표값
    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line

def mean_line(line, lines):     # ((x1,y1),(x2,y2)) 형식으로 바꿔주는 함수인 것 같은데, 왜 있는지 모르겠음(없어도 동작함)
    if line is not None:
        lines.append(line)
    if len(lines) > 0:
        line = np.mean(lines, axis=0, dtype=np.int32)       # 열을 따라 산술평균 구함
        # make sure it's tuples not numpy array for cv2.line to work
        line = tuple(map(tuple, line))

    return line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=15):     # 파란색으로 line 그리기
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return line_image

QUEUE_LENGTH = 50

video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break
    # line_frame = frame
    # select white yellow
    white_yellow = select_white_yellow(frame)       # input : webcam 영상/output : while/yellow 부분만 남김
    gray = cv2.cvtColor(white_yellow, cv2.COLOR_RGB2GRAY)       # grayscale로 변환
    smooth_gray = cv2.GaussianBlur(gray, (15, 15), 0)       # Gaussian smoothing
    edges = cv2.Canny(smooth_gray, 15, 150)     # Edge detection
    regions = select_region_line(edges)      # 관심구역 설정
    # cv2.HoughLinesP(input image(1 channel binary scale), 거리 측정 해상도, 각도(rad), 직선으로 판단할 최소한의 동일 개수, line_length_min, line_length_max)
    # output : 양끝 좌표값 [x1,y1,x2,y2]
    lines = cv2.HoughLinesP(regions, rho=1, theta=np.pi/180, threshold=20, minLineLength=100, maxLineGap=300)
    if lines is not None:
        line_for_van, line_for_cnt = FilterLines(lines)       # detected line의 좌표를 배열(15) 저장
        VanishingPoint = GetVanishingPoint(line_for_van)        # Vanishing point [x0,y0] 구하기
        if VanishingPoint is not None:
            # print("Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point.")
            cv2.circle(regions, (int(VanishingPoint[0]), int(VanishingPoint[1])), 8, (255, 0, 0), -1)       # 소실점 원으로 표시
            cv2.putText(regions, "x : %d, y : %d" %(int(VanishingPoint[0]), int(VanishingPoint[1])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))        # 소실점 좌표값 출력
        
        left_line, right_line = lane_lines(regions, lines)      # line의 양 끝점 ((x1,y1),(x2,y2)) 좌표값
        #self_left_lines = deque(maxlen=QUEUE_LENGTH)
        #self_right_lines = deque(maxlen=QUEUE_LENGTH)
        #left_line = mean_line(left_line,  self_left_lines)
        #right_line = mean_line(right_line, self_right_lines)

        line_image = draw_lane_lines(frame, (left_line, right_line))        # input image에 최종 line 그리기
        # cv2.addWeighted(frame, 1.0, line_image, 0.95, 0.0)
        cv2.imshow('Add lines', cv2.addWeighted(frame, 1.0, line_image, 0.95, 0.0))
        
        print(line_for_cnt)
        print('\n')

    cv2.imshow('original', frame)
    cv2.imshow('result', regions)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video_capture.release()
cv2.destroyAllWindows()
