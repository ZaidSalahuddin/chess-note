import numpy as np
import cv2
from PIL import ImageGrab

bbox = (0,0,1920,1200)

# function to screen shot
def screenshot():
    image = np.array(ImageGrab.grab(bbox=bbox))
    return image

#process to isolate the chessboard from screenshot
def get_board(screen):
    #convert to greyscale
    img = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    #edge detection
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 50, 150)

    #masks
    white_mask = np.all(input == 255, 2).astype(np.uint8)*255  # cv2.inRange(input, (255, 255, 255), (255, 255, 255))
    black_mask = np.all(input == 230, 2).astype(np.uint8)*255  # gray_mask = cv2.inRange(input, (230, 230, 230), (230, 230, 230))

    return img

print('working')
screen = screenshot()
board = get_board(screen)
cv2.imshow("screenshot",board)
cv2.waitKey(0)
cv2.destroyAllWindows()