import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

import random
import numpy as np
import time
import pygame
import sys


# Initialize game
pygame.init()

# Variables
# For Game
speed = 10
speedIncrement = 0
score = 0
startTime = time.time() # starts time countdown
totalTime = 50

# For camera/game window
width, height = 1280, 720

# Starts the game window
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Arcelor Hat Game")

# Initialize Clock for FPS
fps = 30
clock = pygame.time.Clock()

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)  # 3 is an id for the property to set width
cap.set(4, height)  # 4 is an id for a property to set height


# Loading Images
# FireBall
imgFireBall = pygame.image.load('./Resources/fireball.png').convert_alpha()
rectFireBall = imgFireBall.get_rect()
# First place for the fireball to appear
rectFireBall.x, rectFireBall.y = 10, 0

# Hardhat
imgPNG = cv2.imread('./Resources/hardhat.png',cv2.IMREAD_UNCHANGED)
h_png, w_png, c_png = imgPNG.shape
ratio_h_w_png = h_png / w_png

# Initialize the FaceDetector object
# Use MdelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=1)

def resetFireBall():
    rectFireBall.x = random.randint(100, img.shape[1] - 100) # Random place between 100 and shape - 100 to not get the borders
    rectFireBall.y = 0 # Starts at the top

# Starts the main loop for game and capture
start = True
while start:
    # Get Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            start = False
            pygame.quit()
 
    # Countdown for game time
    timeRemain = int(totalTime -(time.time()-startTime))
    if timeRemain <= 0:
        start = False
        pygame.quit()
    else:
        # Start video capture
        success, img = cap.read()
        img = cv2.flip(img, 1) # flip camera in x (1)
        img, bboxs = detector.findFaces(img, draw=False)

        rectFireBall.y += speed  # Move the balloon down
        # check if balloon has reached the bottom of the windown without pop
        if rectFireBall.y > height:
            resetFireBall()
            speed += speedIncrement

        if bboxs:
            for bbox in bboxs:
                # Get face coordinates
                center = bbox["center"]
                x, y, w, h = bbox['bbox']
                score_face_rec = int(bbox['score'][0] * 100)

                # Draw box and recognition score for debug
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
                cvzone.putTextRect(img, f'{score_face_rec}%', (x, y + h + 50))
                cvzone.cornerRect(img, (x, y, w, h))    


                # Set hardhat size                
                new_hat_size_w = int(w * 1.2)
                new_hat_size_h = int (ratio_h_w_png * w)
          
                resized_image = cv2.resize(imgPNG, (new_hat_size_w, new_hat_size_h))
                
                
                # Puts hardhat over the head
                x_hat = int(x+(w-new_hat_size_w)/2)
                y_hat = y-int(w*0.6)                
                imgOverlay = cvzone.overlayPNG(img, resized_image, pos=[x_hat, y_hat])

                # Gets the rectangle of hat and fireball to see collision event for points
                hat_rect = pygame.Rect(x_hat, y_hat, new_hat_size_w,  50)
                ball_rec = pygame.Rect(rectFireBall.x, rectFireBall.y, rectFireBall.w,  rectFireBall.h)
                collide = pygame.Rect.colliderect(hat_rect, ball_rec)
                
                if collide:
                    resetFireBall()
                    score += 10
                    speed += speedIncrement
                    

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgRGB = np.rot90(imgRGB)
            frame = pygame.surfarray.make_surface(imgRGB).convert()
            frame = pygame.transform.flip(frame, True, False)
            window.blit(frame, (0, 0))
            window.blit(imgFireBall, rectFireBall)

            # Prints time and score in the window
            font = pygame.font.Font('./Resources/Marcellus-Regular.ttf', 50)
            textScore = font.render(f'Score: {score}', True, (50, 50, 255))
            textTime = font.render(f'Time: {timeRemain}', True, (50, 50, 255))
            window.blit(textScore, (35, 35))
            window.blit(textTime, (1000, 35))
        
        cv2.waitKey(1) # continuously display frame without needing a button
        # Update Display
        pygame.display.update()
        # Set FPS
        clock.tick(fps)