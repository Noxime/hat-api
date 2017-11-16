import numpy as np
import cv2
import sys
import random

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

orig_orig_img = cv2.imread(sys.argv[1])

use_eyes = False
if "-e" in sys.argv or "--eyes" in sys.argv:
    use_eyes = True

eye_orig = cv2.imread("eye.png", -1)
eye_mask = eye_orig[:, :, 3]
eye_inv = cv2.bitwise_not(eye_mask)

orig_img = cv2.resize(orig_orig_img, (1024, int(1024*(orig_orig_img.shape[0]/orig_orig_img.shape[1]))))
(orig_width, orig_height) = orig_img.shape[:2]

border = orig_width+orig_height

img = cv2.copyMakeBorder(orig_img, border, border, border, border, cv2.BORDER_CONSTANT, value=[0,0,0])
(width, height) = img.shape[:2]


hat = cv2.imread("hat2.png", -1)

#Create masks for compositing
hat_mask = hat[:, :, 3]
hat_inv = cv2.bitwise_not(hat_mask)

#And now lets convert our hat to a 3 component BGR
hat = hat[:, :, 0:3]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    x_ = x
    y_ = y
    w_ = w
    h_ = h
    
    #cv2.rectangle(img,(x, y),(x+w,y+h),(255,0,0),2)
    #cv2.circle(img, (x, y), 5, (0, 255, 0), 2)
    #cv2.circle(img, (x+w, y+h), 5, (0, 0, 255), 2)
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    sizeA = 0
    sizeB = 0

    #Find if we should flip hat based on eye size
    if(len(eyes) >= 2):
        #Oh woops eyes are in wrong order, flip em
        if(eyes[0][0] > eyes[1][0]):
            eye = eyes[0]
            eyes[0] = eyes[1]
            eyes[1] = eye
            
        sizeA = eyes[0][2] * eyes[0][3]
        sizeB = eyes[1][2] * eyes[1][3]

    

    w = int(w * 1.5)
    h = int(h * 1.5)
    
    #Scale hat to proper size for this face
    hat2 = cv2.resize(hat, (w, h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(hat_mask, (w, h), interpolation=cv2.INTER_AREA)
    inv = cv2.resize(hat_inv, (w, h), interpolation=cv2.INTER_AREA)

    ori = -1
    #Slighty translate the hat and flip if necessary
    if(sizeA >= sizeB):
        hat2 = cv2.flip(hat2, 1)
        mask = cv2.flip(mask, 1)
        inv  = cv2.flip(inv , 1)
        ori = -ori

    #Here we do translation of the hat. For some reason
    # X and Y have switched places and I have no idea
    # why.

    xo = -int(h/2.4)

    yo = -int(w/4)
    if(ori < 0):
        yo = 0
    
    #Swap if image is horizontal
    # ???
    o = xo
    xo = yo
    yo = o

    y += yo
    x += xo * ori
        
    nx = max(x, 0)
    mx = min(x+w, width)
    ny = max(y, 0)
    my = min(y+h, height)

    inv = inv[:my-ny, :mx-nx]
    mask = mask[:my-ny, :mx-nx]

    #print((nx, mx), (ny, my))
    
    roi = img[ny:my, nx:mx]

    bg = cv2.bitwise_and(roi, roi, mask=inv)
    fg = cv2.bitwise_and(hat2, hat2, mask=mask)
    
    dst = cv2.add(bg, fg)
    
    img[ny:my, nx:mx] = dst

    for (ex, ey, ew, eh) in eyes:
        ex += x_
        ey += y_
        eh = int(eh * eye_orig.shape[0] / eye_orig.shape[1])

        roi = img[ey:ey+eh, ex:ex+ew, :3]
        
        eye = cv2.resize(eye_orig, (ew, eh))
        mask = cv2.resize(eye_mask, (ew, eh))
        inv = cv2.resize(eye_inv, (ew, eh))

        bg = cv2.bitwise_and(roi, roi, mask=inv)[:, :, :3]
        fg = cv2.bitwise_and(eye, eye, mask=mask)[:, :, :3]

        dst = cv2.add(bg, fg)
        
        img[ey:ey+eh, ex:ex+ew] = dst
    
    print("Pass OK")

    #cv2.rectangle(img, (nx, ny), (mx, my), (255, 255, 0), 2)
    #cv2.circle(img, (nx, ny), 5, (255, 0, 255), 2)
    #cv2.circle(img, (mx, my), 5, (0, 255, 255), 2)

#cv2.circle(img, (0, 0), 5, (255, 0, 0), 2)
#cv2.circle(img, (height, width), 5, (0, 255, 0), 2)

if(len(faces) == 0):
    print("NO FACES")
    sys.exit(0)

#img = img[orig_height:orig_height*2, orig_width:orig_width*2]
img = img[border:-border, border:-border]

#print(orig_height, orig_width)


#cv2.imshow('img',img)
#cv2.waitKey(0)
cv2.imwrite("result.png", img)
#cv2.destroyAllWindows()
