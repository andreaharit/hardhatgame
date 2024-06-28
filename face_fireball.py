import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

import random
import numpy as np
import time
import pygame
import sys


def debug_box_face (debug = False):
    # Draw box and recognition score for debug
    if debug:        
        cvzone.putTextRect(img, f'{score_face_rec}%', (x, y + h + 50))
        cvzone.cornerRect(img, (x, y, w, h))  

def resetFireBall():
    # Restart the fireball if there was a collision    
    rectFireBall.x = random.randint(100, img.shape[1] - 100) # Random place between 100 and shape - 100 to not get the borders
    rectFireBall.y = 0 # Starts at the top

def display_score(score, font):
        textScore = font.render(f'Score: {score}', True, (50, 50, 255)) # text, antialias, color
        window.blit(textScore, (35, 35)) # what to display, (x,y) coordinates in the window

def display_time(timeRemain, font, pause):
    if pause == False:
        textTime = font.render(f'Time: {timeRemain}', True, (50, 50, 255)) 
        
    else:
        textTime = font.render(f'Pause: {timeRemain}', True, (50, 50, 255)) 
    window.blit(textTime, (width - 230, 35))
    



def load_video():
    pass


# Initialize game
pygame.init()

# Variables

hat_dectected = True

# For Game
speed = 10
speedIncrement = 0
score = 0
startTime = time.time() # starts time countdown
camera_delay = 3
game_time = 40
totalTime = game_time + camera_delay
debug = True
pause = False

# For camera/game window
#width, height = 1280, 720 # Full screen
width, height = 800, 600

if hat_dectected:
    # Starts the game window
    window = pygame.display.set_mode((width, height))

    # Title and logo of the window
    pygame.display.set_caption("Arcelor Hat Game")
    icon = pygame.image.load('./Resources/arcelor_logo.png')
    pygame.display.set_icon(icon)

    # Initialize Clock for FPS
    fps = 30
    clock = pygame.time.Clock()

    # Initialize the webcam, capt_dshow makes it quickier to open
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

    # Starts the main loop for game and capture
    start = True

    while start:
        # Countdown for game time
        if pause == False:
            timeRemain = int(totalTime - (time.time()-startTime))
        else:
            totalTime = timeRemain
            startTime = time.time()
        # Get Events
        for event in pygame.event.get():
            # Key controls
            if event.type == pygame.QUIT: # if user clicked the close window, stop the game
                start = False
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:                
                    paused (time= timeRemain)
                elif event.key == pygame.K_x: # exit game pressing X               
                    start = False
                    pygame.quit()
                elif event.key == pygame.K_r: # reset score and time pressing R
                    pause = False
                    totalTime = game_time + 1
                    startTime = time.time()
                    score = 0
                elif event.key == pygame.K_p: # pause time and balls pressing P
                    pause = True
                elif event.key == pygame.K_u: # pause time and balls pressing P
                    pause = False
        
        if timeRemain <= 0:
            start = False
            pygame.quit()
        else:
            # Start video capture
            success, img = cap.read()
            img = cv2.flip(img, 1) # flip camera in x (1)
            img, bboxs = detector.findFaces(img, draw=False)

            rectFireBall.y += speed  # Move the fireball down
            # check if fireball  has reached the bottom of the windown without pop
            if rectFireBall.y > height:
                if pause == False:
                    resetFireBall()
                    speed += speedIncrement
            # if a face is detected!
            if bboxs:
                for bbox in bboxs:
                    # Get face coordinates
                    center = bbox["center"]
                    x, y, w, h = bbox['bbox']
                    score_face_rec = int(bbox['score'][0] * 100)

                    # Draw box and recognition score for debug
                    debug_box_face (debug = debug)


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
                    
                    if collide and pause == False:
                        resetFireBall()
                        score += 10
                        speed += speedIncrement
                        

                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert camera capture BGR to RGB because of the screen
                imgRGB = np.rot90(imgRGB)
                frame = pygame.surfarray.make_surface(imgRGB).convert() # reconverts the rotated image to a surface in pygame
                frame = pygame.transform.flip(frame, True, False) # flip all again, it's something to do with the cv2
                window.blit(frame, (0, 0))
                window.blit(imgFireBall, rectFireBall)

                # Prints time and score in the window
                font = pygame.font.Font('./Resources/Marcellus-Regular.ttf', 50)
                display_score(font = font, score = score)
                display_time (font= font, timeRemain=timeRemain, pause= pause)
            
            cv2.waitKey(1) # continuously display frame without needing a button
            # Update Display
            pygame.display.update()
            # Set FPS
            clock.tick(fps)

# Put another 3 types of things falling
# Record the score for each thingy
# Use the score of the biggest one to get a video of a training