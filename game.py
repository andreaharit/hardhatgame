import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

import random
import numpy as np
import time
import pygame
from pathlib import Path


# Variables
DEBUG = False  # Do you want to see the bbox of face detection with the detection score?

# Game variables
GAME_TIME = 5  # Seconds to be played
CAMERA_DELAY = 1  # Delay correction for camera starting
TOTAL_TIME = GAME_TIME + CAMERA_DELAY
FIREBALL_SPEED = 10  # How fast the fireball goes down
SCALE = 0.6  # Scale (float), to scale the size of the fireball up or down
SCORE_LIMIT = 10  # If score <= this one, display first video, if > then the other video
COLOR_LETTERS = (255, 56, 1)  # Color of the score and time letters
BACKGROUND_BOXES = (
    255,
    255,
    255,
)  # Color for the background boxes of the score and time

# Camera and window variables
FPS = 30  # FPS for the game
# Size of windows to be opened. Use 1280, 720 for a full screen. 800, 600 for tests
FULL_SCREEN = True
SCREEN_WIDTH, SCREEN_HEIGHT = (
    1280,
    660,
)
CAMERA_NUMBER = 0  # Set 0 is the webcam, others might be another number
CAMERA_RANGE = 1  # Set 0 for if person is close to camera, 1 if far

# File paths
ROOT = Path(__file__).parent

# Directory Structure
GAME_RESOURCES_DIR = ROOT / "Resources/Game"
VIDEO_RESOURCES_DIR = ROOT / "Resources/Videos"

# Files
IMG_ICON = GAME_RESOURCES_DIR / "arcelor_logo.png"
IMG_HARDHAT = GAME_RESOURCES_DIR / "hardhat.png"
FONT = GAME_RESOURCES_DIR / "VAG-Rounded-Regular.ttf"

VIDEOS = {
    0: VIDEO_RESOURCES_DIR / "1-Comora.mp4",
    1: VIDEO_RESOURCES_DIR / "6-Guaranteed.mp4",
}

# Game categories for the image of the fireball
CATEGORIES = {
    "DEFAULT": GAME_RESOURCES_DIR / "fireball.png",
    "RH": GAME_RESOURCES_DIR / "RH.png",
    "FINANCE": GAME_RESOURCES_DIR / "finance.png",
    "MECHANIC": GAME_RESOURCES_DIR / "tools.png",
    "COKEFRABRIC": GAME_RESOURCES_DIR / "coal.png",
    "IT": GAME_RESOURCES_DIR / "snake.png",
    "TOWERCRANEOP": GAME_RESOURCES_DIR / "crane.png",
}


class Game:
    """
    Contains the code for face detection and the game logic.
    Params:
    - fireball (str): category that will define wich image will be make the fireball.
    """

    def __init__(self, fireball: str) -> None:
        # Initializes game
        pygame.init()
        # Starts the game window
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Sets and display title and logo in the window
        pygame.display.set_caption("Arcelor Hat Game")
        icon = pygame.image.load(IMG_ICON)
        pygame.display.set_icon(icon)

        # Initializes clock for keeping the FPS for the game
        fps = FPS
        clock = pygame.time.Clock()

        # Initializes the webcam cv2.CAP_DSHOW is makes video capture quicker
        cap = cv2.VideoCapture(CAMERA_NUMBER, cv2.CAP_DSHOW)
        cap.set(3, SCREEN_WIDTH)
        cap.set(4, SCREEN_HEIGHT)

        # Loads images and gets their box coordinates
        # FireBall
        img_fireball = pygame.image.load(fireball).convert_alpha()
        img_fireball = pygame.transform.scale_by(img_fireball, SCALE)
        rect_fireball = img_fireball.get_rect()

        # Hardhat
        img_hat = cv2.imread(IMG_HARDHAT, cv2.IMREAD_UNCHANGED)
        height_hat, width_hat, channel_hat = img_hat.shape
        proportion_hat = height_hat / width_hat

        # Initializes the FaceDetector object
        detector = FaceDetector(minDetectionCon=0.5, modelSelection=CAMERA_RANGE)

        # Starts variable settings for game
        speed = FIREBALL_SPEED
        start = True
        pause = False
        total_time = TOTAL_TIME
        start_time = time.time()
        scored = 0  # Catched fireballs
        missed = 0  # Missed fireballs
        total = scored + missed
        # First place for the fireball to appear
        rect_fireball.x, rect_fireball.y = (
            SCREEN_WIDTH / 2,
            0,
        )

        while start:
            # Countdown
            if pause == False:
                time_remain = int(total_time - (time.time() - start_time))
            else:
                total_time = time_remain
                start_time = time.time()
            # Key controls for the game
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    start = False
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Exits game by pressing q
                        start = False
                        pygame.quit()
                    elif event.key == pygame.K_r:  # Resets score and time by pressing r
                        pause = False
                        total_time = (
                            GAME_TIME + 1
                        )  # Small 1s correction because of system delay
                        start_time = time.time()
                        scored = 0
                        missed = 0
                        total = 0
                    elif event.key == pygame.K_p:  # Pauses by pressing P
                        pause = True
                    elif event.key == pygame.K_u:  # Unpauses by pressing P
                        pause = False

            if time_remain <= 0:  # Stops game if time is up
                start = False
                pygame.quit()
            else:
                # GAME LOGIC
                # Start video capture
                success, img = cap.read()
                img = cv2.flip(img, 1)

                # Gets the bbox of detected faces and img
                img, bboxs = detector.findFaces(img, draw=False)
                rect_fireball.y += speed  # Move the fireball down

                # Checks if fireball has reached the bottom to respawn
                if rect_fireball.y > SCREEN_HEIGHT and pause == False:
                    rect_fireball.x, rect_fireball.y = self.reset_fireball()
                    missed += 1

                if bboxs:  # If a face is detected
                    for bbox in bboxs:
                        # Gets face coordinatesm and detection score
                        x, y, w, h = bbox["bbox"]
                        score_face_rec = int(bbox["score"][0] * 100)

                        # Draws box and recognition score for debug
                        self.debug_box_face(
                            img=img, score_face_rec=score_face_rec, x=x, y=y, h=h, w=w
                        )

                        # Set shardhat size
                        new_hat_size_w = int(w * 1.2)
                        new_hat_size_h = int(proportion_hat * w)
                        resized_image = cv2.resize(
                            img_hat, (new_hat_size_w, new_hat_size_h)
                        )

                        # Puts hardhat img over the head with size adjustments
                        x_hat = int(x + (w - new_hat_size_w) / 2)
                        y_hat = y - int(w * 0.8)
                        imgOverlay = cvzone.overlayPNG(
                            img, resized_image, pos=[x_hat, y_hat]
                        )

                        # Gets the rectangle of hat and fireball to set collision event
                        hat_rect = pygame.Rect(x_hat, y_hat, new_hat_size_w, 50)
                        ball_rec = pygame.Rect(
                            rect_fireball.x,
                            rect_fireball.y,
                            rect_fireball.w,
                            rect_fireball.h,
                        )
                        collide = pygame.Rect.colliderect(hat_rect, ball_rec)

                        if collide and pause == False:
                            rect_fireball.x, rect_fireball.y = self.reset_fireball()
                            scored += 1

                        total = scored + missed
                    # Camera to screen corrections
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imgRGB = np.rot90(imgRGB)
                    frame = pygame.surfarray.make_surface(imgRGB).convert()
                    frame = pygame.transform.flip(frame, True, False)
                    self.window.blit(frame, (0, 0))
                    self.window.blit(img_fireball, rect_fireball)

                    # Prints time and score in the window
                    font = pygame.font.Font(FONT, 50)
                    self.display_score(font=font, score=scored, total=total)
                    self.display_time(font=font, time_remain=time_remain, pause=pause)
                # Mantains display of game
                cv2.waitKey(1)
                pygame.display.update()
                clock.tick(fps)
        cap.release()

        self.final_score = scored
        self.missed = missed
        self.total = total

    def debug_box_face(self, img, score_face_rec, x, y, h, w) -> None:
        """
        Draw around the face a box and recognition score for debug.
        """

        if DEBUG:
            cvzone.putTextRect(img, f"{score_face_rec}%", (x, y + h + 50))
            cvzone.cornerRect(img, (x, y, w, h))

    def reset_fireball(self):
        """
        Restart the fireball  in a random place at the top if there was a collision
        """
        rect_fireball_x = random.randint(100, SCREEN_WIDTH - 100)
        rect_fireball_y = 0
        return rect_fireball_x, rect_fireball_y

    def display_score(self, font, score: int, total: int) -> None:
        """
        Displays score.
        """
        textScore = font.render(
            f"Score: {score} of {total}", True, COLOR_LETTERS
        )  # text, antialias, color

        score_rect = textScore.get_rect(topleft=(35, 35))
        self.draw_box(score_rect)

        self.window.blit(
            textScore, (35, 35)
        )  # what to display, (x,y) coordinates in the window

    def display_time(self, font, time_remain: int, pause: bool):
        """
        Displays time count.
        """
        if pause == False:
            textTime = font.render(f"Time: {time_remain}", True, COLOR_LETTERS)

        else:
            textTime = font.render(f"Pause: {time_remain}", True, COLOR_LETTERS)

        time_rec = textTime.get_rect(topleft=(SCREEN_WIDTH - 230, 35))

        self.draw_box(time_rec)
        self.window.blit(textTime, (SCREEN_WIDTH - 230, 35))

    def draw_box(self, text_rect) -> None:
        """
        Generate a filled in box behind. Used to make the score and time count prettier.
        Params:
        - text_rect: coordinates of the box around the text that will be inside the box.
        """
        # Padding
        padding = 10
        # Calculates background rectangle dimensions
        rect_width = text_rect.width + padding * 2
        rect_height = text_rect.height + padding * 2
        rect_x = text_rect.x - padding
        rect_y = text_rect.y - padding

        # Draws the box
        pygame.draw.rect(
            self.window,
            BACKGROUND_BOXES,
            (rect_x, rect_y, rect_width, rect_height),
            border_radius=10,
        )


class Play_Video:
    """
    Class that plays any video via cv2.
    Params:
    - video (path): video to be played after the game.
    """

    def __init__(self, video) -> None:

        cap = cv2.VideoCapture(video)
        window_name = "video"
        cv2.namedWindow(window_name)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        while cap.isOpened():
            ret, frame = cap.read()  # Captures video frames
            if not ret:  #
                break
            frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            cv2.imshow(window_name, frame)
            keyCode = cv2.waitKey(5)  # Keeps a look if any keys were pressed
            # Check for user pressing q or the window to close the video
            if (
                keyCode == ord("q")
                or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
        cap.release()
        cv2.destroyAllWindows()


def game_video(category: str, has_hardhat: bool = False) -> None:
    """
    Main function that wraps up the game and the video play.
    Params:
    - category (str): job area that will change the image for the fireball.
    This category would come from an outside LLM that would identify the area the person wants to work on.
    - has_hardhat (bool): booleand to know if the game will be played or not (person has a hardhat or not).
    This boolean depends on another visual recognition program that will trigger the game.
    """
    if has_hardhat:
        if category in CATEGORIES:
            fireball = CATEGORIES[category]
        else:
            fireball = CATEGORIES["DEFAULT"]

        game = Game(fireball=fireball)  # Starts the game
        score = game.final_score
        # Selects the video based on the score since we don't have videos based on job postings yet
        if score <= SCORE_LIMIT:
            video = VIDEOS[0]
        else:
            video = VIDEOS[1]

        # Plays the video
        playing = Play_Video(video=video)


if __name__ == "__main__":
    # Variables that come from integration with other solutions
    has_hardhat = True # Did the YOLO model detected the hardhat? (will the game start?)
    category = "IT" # What job category the LLM passed foward (which icon will fall?)

    game_video(category=category, has_hardhat=has_hardhat)
