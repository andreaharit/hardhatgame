import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

import random
import numpy as np
import time
import pygame
import sys

# Variables
GAME_TIME = 40 # Seconds to be played
CAMERA_DELAY= 3 # Delay for camera to start
TOTAL_TIME = GAME_TIME + CAMERA_DELAY
FIREBALL_SPEED = 10 # How fast the fireball goes down
DEBUG = True # Do you want to see the bbox of face detection with the detection score?
CAMERA_NUMBER = 0 # 0 is the webcam, others might be another number
CAMERA_RANGE = 0 # 0 for if person is close to camera, 1 if far
FPS = 30
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600 # Set to 1280, 720 for a full screen game

# File paths
IMG_ICON = './Resources/arcelor_logo.png'
IMG_FIREBALL = './Resources/fireball.png'
IMG_HARDHAT = './Resources/hardhat.png'
FONT = './Resources/Marcellus-Regular.ttf'


class Game:
    def __init__ (self):       
        # Initialize game
        pygame.init()
        # Starts the game window
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Title and logo of the window
        pygame.display.set_caption("Arcelor Hat Game")
        icon = pygame.image.load(IMG_ICON)
        pygame.display.set_icon(icon)

        # Initialize Clock for FPS
        fps = FPS
        clock = pygame.time.Clock()

        # Initialize the webcam, capt_dshow makes it quickier to open
        cap = cv2.VideoCapture(CAMERA_NUMBER, cv2.CAP_DSHOW)
        cap.set(3, SCREEN_WIDTH)  
        cap.set(4, SCREEN_HEIGHT)  

        # Loading Images and getting their box coordinates
        # FireBall
        img_fireball = pygame.image.load(IMG_FIREBALL).convert_alpha()
        rect_fireball = img_fireball.get_rect()
        
        # Hardhat
        img_hat = cv2.imread(IMG_HARDHAT,cv2.IMREAD_UNCHANGED)
        height_hat, width_hat, channel_hat = img_hat.shape
        proportion_hat = height_hat / width_hat

        # Initialize the FaceDetector object
        detector = FaceDetector(minDetectionCon=0.5, modelSelection = CAMERA_RANGE)
        
        # Variables for game
        speed = FIREBALL_SPEED
        start = True
        pause = False
        total_time = TOTAL_TIME  
        start_time = time.time() 
        score = 0    
        rect_fireball.x, rect_fireball.y = SCREEN_WIDTH/2, 0 # First place for the fireball to appear

        while start:
            if pause == False:
                time_remain = int(total_time - (time.time() - start_time))
            else:
                total_time = time_remain
                start_time = time.time()
            # Get events for key controls
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    start = False
                    pygame.quit()
                if event.type == pygame.KEYDOWN:       
                    if event.key == pygame.K_x: # exit game, pressing X               
                        start = False
                        pygame.quit()
                    elif event.key == pygame.K_r: # reset score and time, pressing R
                        pause = False
                        total_time = GAME_TIME + 1 # small 1s correction
                        start_time = time.time()
                        score = 0
                    elif event.key == pygame.K_p: # pause time and balls pressing P
                        pause = True
                    elif event.key == pygame.K_u: # pause time and balls pressing P
                        pause = False
            
            if time_remain <= 0:
                start = False
                pygame.quit()
            else:
                # Start video capture
                success, img = cap.read()
                img = cv2.flip(img, 1)
                img, bboxs = detector.findFaces(img, draw=False)
                rect_fireball.y += speed  # Move the fireball down
                # check if fireball  has reached the bottom of the windown without pop
                if rect_fireball.y > SCREEN_HEIGHT: # respaw if it reachs botton
                    if pause == False:
                        rect_fireball.x, rect_fireball.y = self.reset_fireball()
                
                if bboxs: # if a face is detected
                    for bbox in bboxs:
                        # Get face coordinates
                        center = bbox["center"]
                        x, y, w, h = bbox['bbox']
                        score_face_rec = int(bbox['score'][0] * 100)

                        # Draw box and recognition score for debug
                        self.debug_box_face(img = img, score_face_rec = score_face_rec, x = x, y = y, h = h, w = w)

                        # Set hardhat size                
                        new_hat_size_w = int(w * 1.2)
                        new_hat_size_h = int (proportion_hat * w)
                
                        resized_image = cv2.resize(img_hat, (new_hat_size_w, new_hat_size_h))
                        
                        
                        # Puts hardhat over the head
                        x_hat = int(x+(w-new_hat_size_w)/2)
                        y_hat = y-int(w*0.6)                
                        imgOverlay = cvzone.overlayPNG(img, resized_image, pos=[x_hat, y_hat])

                        # Gets the rectangle of hat and fireball to see collision event for points
                        hat_rect = pygame.Rect(x_hat, y_hat, new_hat_size_w,  50)
                        ball_rec = pygame.Rect(rect_fireball.x, rect_fireball.y, rect_fireball.w,  rect_fireball.h)
                        collide = pygame.Rect.colliderect(hat_rect, ball_rec)
                        
                        if collide and pause == False:
                            rect_fireball.x, rect_fireball.y = self.reset_fireball()
                            score += 10                            

                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert camera capture BGR to RGB because of the screen
                    imgRGB = np.rot90(imgRGB)
                    frame = pygame.surfarray.make_surface(imgRGB).convert() # reconverts the rotated image to a surface in pygame
                    frame = pygame.transform.flip(frame, True, False) # flip all again, it's something to do with the cv2
                    self.window.blit(frame, (0, 0))
                    self.window.blit(img_fireball, rect_fireball)

                    # Prints time and score in the window
                    font = pygame.font.Font(FONT, 50)
                    self.display_score(font = font, score = score)
                    self.display_time (font = font, time_remain = time_remain, pause= pause)
                
                cv2.waitKey(1) # continuously display frame without needing a button
                # Update Display
                pygame.display.update()
                # Set FPS
                clock.tick(fps)

    def debug_box_face(self, img, score_face_rec, x,y,h,w):
        # Draw box and recognition score for debug
        if DEBUG:        
            cvzone.putTextRect(img, f'{score_face_rec}%', (x, y + h + 50))
            cvzone.cornerRect(img, (x, y, w, h))  

    def reset_fireball(self):
        # Restart the fireball if there was a collision in a random place at the top  
        rect_fireball_x = random.randint(100, SCREEN_WIDTH - 100) 
        rect_fireball_y = 0 
        return rect_fireball_x, rect_fireball_y
    
    def display_score(self, font, score):
        # Displays score
        textScore = font.render(f'Score: {score}', True, (50, 50, 255)) # text, antialias, color
        self.window.blit(textScore, (35, 35)) # what to display, (x,y) coordinates in the window

    def display_time(self, font, time_remain, pause):
        # Displays time
        if pause == False:
            textTime = font.render(f'Time: {time_remain}', True, (50, 50, 255))             
        else:
            textTime = font.render(f'Pause: {time_remain}', True, (50, 50, 255)) 
        self.window.blit(textTime, (SCREEN_WIDTH - 230, 35))        

    

game = Game()               
