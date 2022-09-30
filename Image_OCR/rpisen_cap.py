###9/29 기준, 제대로 돌아가는 코드


import cv2
from PIL import Image
import pytesseract
import time
import ripsensor
import RPi.GPIO as GPIO   
 
#GPIO 핀 17,27을 사용한다.
GPIOIN = 17
GPIOOUT = 27
 
#핀 넘버링을 BCM 방식을 사용한다.
GPIO.setmode(GPIO.BCM)   
print "HC-SR501 motion detection start"
#17번 핀을 입력용, 27번 핀을 출력용으로 설정한다.
#출력용 핀은 LED 상태를 확인하기 위해 사용하는 핀으로 실제 동작과는 무관하다.
GPIO.setup(GPIOIN, GPIO.IN)   
GPIO.setup(GPIOOUT, GPIO.OUT)   

def ri():
    while True:
        if GPIO.input(pinmode) == GPIO.LOW:
            print("BOX LOAD")
            a = 10
            return a
        else:
            print("NO BOX")
            b = 20
            return b
        time.sleep(0.2)

cap = cv2.VideoCapture(0)  # 0: default camera

while cap.isOpened():
    # 카메라 프레임 읽기
    success, frame = cap.read()
    if success:
        # 프레임 출력
        cv2.imshow('Camera Window', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        ret, bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        img = Image.fromarray(bin)
        
        # ESC를 누르면 캡처
        condition = ri()
        if (condition == 10):
            return_value, image = cap.read()
            cv2.imwrite("opencv.jpg", image)
            time.sleep(1)
            
            pytesseract.pytesseract.tesseract_cmd = r'C:/Tesseract-OCR/tesseract.exe'
            text = pytesseract.image_to_string(Image.open("opencv.jpg"), lang="eng")

            print(text.replace(" ", ""))
            #소켓으로 다시 라즈베리로 보내는 코드임
            
            break




cap.release()
cv2.destroyAllWindows()
