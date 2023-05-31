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
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # blurs image
    blur = cv2.medianBlur(gray, 5)
    #show the blur 
    cv2.imshow("blur",blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #thresholds image
    ret, thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
    #show the thresh 
    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #downscales image
    # scale_percent = 60 # percent of original size
    # width = int(screen.shape[1] * scale_percent / 100)
    # height = int(screen.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)

    #finds chessboard corners
    #code just to find best corner dimensions
    numbers = [5,6,7,8,9,10,11]
    for i in numbers:
        for x in numbers:
            #sets the dimension of the chessboardcorners function
            corner_dim = (i,x)
            ret, corners = cv2.findChessboardCorners(thresh, corner_dim,None) 
            print(corners)
            print('ret: ', ret)

            #the mask for the image
            mask = np.zeros_like(screen)

            #draw the chessboard area on the mask
            mask = cv2.drawChessboardCorners(mask, corner_dim, corners, ret)
            #drawing a rectagle over it igb
            #mask = cv2.rectangle(mask, corners[0],corners[1])
            
            #show the mask 
            cv2.imshow(f"mask {i} by {x}",mask)

            #rescale the mask
            # new_dim = (int(screen.shape[1]), int(screen.shape[0]))
            # resized_mask = cv2.resize(mask, new_dim, interpolation=cv2.INTER_AREA)
            resized_mask = mask

            #appply mask
            board = cv2.bitwise_or(screen,resized_mask)

            #show the board 
            cv2.imshow("board",board)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    return board

print('working')
screen = screenshot()
print('screen shotted')
board = get_board(screen)
print('board processed')

cv2.imshow("screenshot",screen)
cv2.imshow("board",board)
cv2.waitKey(0)
cv2.destroyAllWindows()