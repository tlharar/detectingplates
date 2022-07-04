
#import ossaudiodev

import cv2
import imutils
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

image = cv2.imread(r'C:\Users\tlhar\Desktop\araba\images (5).jpg')
finding_cars = cv2.CascadeClassifier(r'C:\Users\tlhar\Desktop\araba\cas4.xml')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_image,(5,5),0)
dilated = cv2.dilate(blur,np.ones((2,2)))
cars = finding_cars.detectMultiScale(dilated, 1.1, 2)
img2 = image.copy()
for (x,y,w,h) in cars:
    cv2.rectangle(img2, (x,y), (x+w,y+h), (0,0,255), 2)

kesit = image[y:y+h,x:x+w]

img = kesit
img = imutils.resize(img, width=300)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_img = cv2.bilateralFilter(gray_img, 11 , 17, 17)

edge = cv2.Canny(gray_img, 30, 200)

cnts, new = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]


screenCnt = None


i = 7
for c in cnts:
    perimeter = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4:
        screenCnt = approx
    x,y,w,h = cv2.boundingRect(c)
    new_img = img[y:y+h,x:x+w]
    cv2.imwrite('./' + str(i)+'.png', new_img)
    i+=1
    break
    
   
if screenCnt is None:
    detected = 0 
    print("PLATE CAN'T DETECTED")

else:
    detected = 1
if detected == 1:
    cv2.drawContours(img ,[screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("image with detected license plate", img)
cv2.waitKey(0)

cut_plate = './7.png'


cv2.imshow("cut", cv2.imread(cut_plate))
if detected ==1:
    text = pytesseract.image_to_string(cut_plate)
    print("Licance Plate : ", text)

cv2.waitKey(0)
cv2.destroyAllWindows()
