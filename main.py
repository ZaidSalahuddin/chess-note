import numpy as np
import cv2
from PIL import ImageGrab

bbox = (0,0,1600,1200)

# function to screen shot
def screenshot():
    image = np.array(ImageGrab.grab(bbox=bbox))
    return image

def get_board():
    pass

print('working')
screen = screenshot()
cv2.imshow("screenshot",screen)
cv2.waitKey(0)
cv2.destroyAllWindows()