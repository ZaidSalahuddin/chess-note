import numpy as np
import cv2
from PIL import ImageGrab

bbox = (0,0,1920,1200)

#function for distance formula
def distance(x1,y1,x2,y2):
    x_diff = x2-x1
    y_diff = y2-y1
    x_diff_sq = np.abs(x_diff^2)
    y_diff_sq = np.abs(y_diff^2)
    dist = np.sqrt(x_diff_sq+y_diff_sq)
    return dist

# function to screen shot
def screenshot():
    image = np.array(ImageGrab.grab(bbox=bbox))
    return image

#process to isolate the chessboard from screenshot
def get_board(screen):
    #convert to greyscale
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # blurs image
    blur = cv2.medianBlur(gray, 7)
    blur = cv2.medianBlur(blur, 5)
    #show the blur 
    cv2.imshow("blur",blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #thresholds image
    ret, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    #show the thresh 
    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #inverted image
    inv = cv2.bitwise_not(thresh)
    #inverted imageb
    cv2.imshow("inverted",inv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #finds chessboard corners
    #code just to find best corner dimensions
    #sets the dimension of the chessboardcorners function
    corner_dim = (7,7)
    ret, corners = cv2.findChessboardCorners(inv, corner_dim,None) 

    #the mask for the image
    mask = np.zeros_like(screen)
    
    if ret == True:
        #draw the chessboard area on the mask
        #mask = cv2.drawChessboardCorners(mask, corner_dim, corners, ret)
        
        #extend the mask to the 8x8 sides
        #use corners 0 and 8 for top right square
        #use index -1 and -9 for bottom left
        #print("corners shape",corners)
        #makes the array 2d from 3d
        corners_2d = corners
        corners_2d = np.reshape(corners_2d,(49,2))

        #gets the distance from the corners of the corner squares and finds a point 1 diagonal away
        side_length_top = distance(int(corners_2d[0,0]), int(corners_2d[0,1]), int(corners_2d[8,0]), int(corners_2d[8,1]))
        side_length_bottom = distance(int(corners_2d[-1,0]), int(corners_2d[-1,1]), int(corners_2d[-9,0]), int(corners_2d[-9,1]))

        top_point = (int(corners_2d[0,0]-side_length_top), int(corners_2d[0,1]-side_length_top))
        bottom_point = (int(corners_2d[-1,0]-side_length_bottom), int(corners_2d[-1,1]-side_length_bottom))

        #drawing a rectagle mask with the previous point
        print(gray.shape)
        rect = np.zeros(gray.shape, dtype=np.uint8)
        mask = np.reshape(rect, rect.shape + (1,))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.rectangle(mask,top_point,bottom_point,255)


        print("mnask shape: ", mask.shape)

        #show the mask 
        cv2.imshow(f"mask",mask)

        #appply mask
        board = cv2.bitwise_or(screen,mask)

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