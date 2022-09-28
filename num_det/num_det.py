import pytesseract
import cv2
import numpy as ny
import re#추가
import threading#추가
import time#추가
import collections
from collections import Counter

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN


# pytesseract.pytesseract.tesseract_cmd = "C:/Tesseract-OCR/tesseract.exe"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = VideoCapture(1)#0번=> 외부 웹캠/ 1번이 컴터캠 
if not cap.isOpened():
    raise IOError('Can not read webcam...')

cntr = 0
numbers=[]#list()#인식된 값중 숫자만 추출한 값들
num=[]#list()#숫자들 들어있는 리스트
#new_num=list()#새로 들어온 원소 리스트
#again_num = list()#중복된 원소 리스트
#new_again_num = list()#중복된 원소들 중 한번만 중복인 애들
trust_num = []#list()
#max_item = list()
count_b = {}#딕셔너리

# 비디오 FPS 지정
#prev_time = 0
#FPS = 30

while True:
    ret, frame = cap.read() # ret : 프레임 읽기를 성고하면 True 값 변환, frame : 배열 형식의 영상 프레임 (가로 X 세로 X 3) 값 반환
    #current_time = time.time() - prev_time
    #cntr += 1
    #if((cntr%30) == 0):#프레임30개마다로 바꿈
    #if(ret is True) and (current_time > 1./FPS):
    if not ret:    
        imgH, imgW, _ = frame.shape
        x1, y1, w1, h1 = 0, 0, imgH, imgW
        imgboxes = pytesseract.image_to_boxes(frame)
        
        config = ('--oem 1 --psm 7')
        imgchar3 = pytesseract.image_to_string(frame, config=config)
        print(imgchar3)
        
        #숫자만 뽑아내기
        print('==========================')
        string = imgchar3
        numbers = re.findall("\d+",string)
        print('numbers=',numbers)
        
        #중복 값 체크
        cnt = Counter(numbers)
        trust_num=cnt.most_common(3)
        
        for boxes in imgboxes.splitlines():
            boxes = boxes.split(' ')
            x,y,w,h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
            cv2.rectangle(frame, (x, imgH-y), (w, imgH-h), (255,0,0), 3)

        cv2.putText(frame, imgchar3, (x1+int(w1/50), y1+int(h1/20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.imshow('Video detection ', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
                  print('trust_num=',trust_num)
                  break
                
#몇초동안 같은값이 인식되면 그 값을 최종값에 저장해서  출력
    
cap.release()
cv2.destroyAllWindows()
