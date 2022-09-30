from collections import deque
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob     # 운영체제와의 상호작용, 파일들의 리스트를 뽑을때 사용
from moviepy.editor import VideoFileClip    # 비디오 처리
import math
# Threshold by which lines will be rejected wrt the horizontal
REJECT_DEGREE_TH = 4.0

def select_white_yellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # white color mask
    lower = np.uint8([0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([10,   0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        # in case, the input image has a channel dimension
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
    return cv2.bitwise_and(image, mask)

def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols*0.1, rows*0.95]
    top_left = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right = [cols*0.6, rows*0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image)  # don't want to modify the original
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        if line is not None:
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
    left_lane = np.dot(left_weights,  left_lines) / \
        np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / \
        np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    # print("left : ", left_lane)
    # print("right : ", right_lane, "\n")

    y1 = image.shape[0]  # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def FilterLines(Lines):
    FinalLines = []

    for Line in Lines:
        if Line is not None:
            [[x1, y1, x2, y2]] = Line

            # Calculating equation of the line: y = mx + c
            if x1 != x2:
                m = (y2 - y1) / (x2 - x1)
            else:
                m = 100000000
            c = y2 - m*x2
            # theta will contain values between -90 ~ +90.
            theta = math.degrees(math.atan(m))

            # Rejecting lines of slope near to 0 degree or 90 degree and storing others
            if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
                l = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)    # length of the line
                FinalLines.append([x1, y1, x2, y2, m, c, l])

    # Removing extra lines
    # (we might get many lines, so we are going to take only longest 15 lines
    # for further computation because more than this number of lines will only
    # contribute towards slowing down of our algo.)
    if len(FinalLines) > 15:
        # x[-1] : FinalLines의 제일 마지막 값 = l, reverse = True : 내림차순 -> 길이로 내림차순
        FinalLines = sorted(FinalLines, key=lambda x: x[-1], reverse=True)
        FinalLines = FinalLines[:15]

    return FinalLines

def GetVanishingPoint(Lines):
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

QUEUE_LENGTH = 50

class LaneDetector:
    def __init__(self):
        self.left_lines = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def process(self, image):
        white_yellow            = select_white_yellow(image)
        gray                    = cv2.cvtColor(white_yellow, cv2.COLOR_RGB2GRAY)
        smooth_gray             = cv2.GaussianBlur(gray, (15, 15), 0)     # Kernel size = 15
        edges                   = cv2.Canny(smooth_gray, 15, 150)
        regions                 = select_region(edges)
        lines                   = cv2.HoughLinesP(regions, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
        line_for_van            = FilterLines(lines)
        VanishingPoint          = GetVanishingPoint(line_for_van)
        left_line, right_line   = lane_lines(image, lines)

        if VanishingPoint is None:
            print("Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point.")
            
        cv2.circle(image, (int(VanishingPoint[0]), int(VanishingPoint[1])), 8, (255, 0, 0), -1)
        cv2.putText(image, "x : %d, y : %d"%(int(VanishingPoint[0]), int(VanishingPoint[1])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        
        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines) > 0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                # make sure it's tuples not numpy array for cv2.line to work
                line = tuple(map(tuple, line))
            return line

        left_line = mean_line(left_line,  self.left_lines)
        right_line = mean_line(right_line, self.right_lines)

        return draw_lane_lines(image, (left_line, right_line))

def process_video(video_input, video_output):
    detector = LaneDetector()

    clip = VideoFileClip(os.path.join('C:/Users/yoosk/Documents/GitHub/Embedded-SW/Line_Detect/test_videos', video_input))
    processed = clip.fl_image(detector.process)
    processed.write_videofile(os.path.join('C:/Users/yoosk/Documents/GitHub/Embedded-SW/Line_Detect/output_videos', video_output), audio=False)

process_video('solidWhiteRight.mp4', 'white.mp4')
process_video('solidYellowLeft.mp4', 'yellow.mp4')
process_video('challenge.mp4', 'extra.mp4')
