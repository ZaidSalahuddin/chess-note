import numpy as np
import cv2
from PIL import ImageGrab
from PIL import Image
from board_to_fen.predict import get_fen_from_image

bbox = (0,0,1920,1200)

#function for distance formula just becoz
def distance(x1,y1,x2,y2):
    x_diff = x2-x1
    y_diff = y2-y1
    x_diff_sq = x_diff**2
    y_diff_sq = y_diff**2
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

    #thresholds image
    ret, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    
    #inverted image
    inv = cv2.bitwise_not(thresh)

    #finds chessboard corners
    #code just to find best corner dimensions
    #sets the dimension of the chessboardcorners function
    corner_dim = (7,7)
    ret, corners = cv2.findChessboardCorners(inv, corner_dim,None) 

    #the mask for the image
    mask = np.zeros_like(screen)
    
    if ret == True:
        #draw the chessboard area on the mask
        mask = cv2.drawChessboardCorners(mask, corner_dim, corners, ret)
        
        #extend the mask to the 8x8 sides
        #use corners 0 and 8 for top right square
        #use index -1 and -9 for bottom left
        #print("corners shape",corners)
        #makes the array 2d from 3d
        corners_2d = corners
        corners_2d = np.reshape(corners_2d,(49,2))

        #gets the distance from the corners of the corner squares and finds a point 1 diagonal away
        side_length_top = distance(int(corners_2d[0,0]), int(corners_2d[0,1]), int(corners_2d[7,0]), int(corners_2d[7,1]))
        side_length_bottom = distance(int(corners_2d[-1,0]), int(corners_2d[-1,1]), int(corners_2d[-8,0]), int(corners_2d[-8,1]))

        top_point = (int(corners_2d[0,0]-side_length_top), int(corners_2d[0,1]-side_length_top))
        bottom_point = (int(corners_2d[-1,0]+side_length_bottom), int(corners_2d[-1,1]+side_length_bottom))

        #drawing a rectagle mask with the previous point
        cv2.rectangle(mask,top_point,bottom_point,(255,255,255),-1)

        print("mnask shape: ", mask.shape)

        #show the mask 
        cv2.imshow(f"mask",mask)

        #appply mask
        board = cv2.bitwise_and(screen,mask)
        board = board[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

        # grayscale the board to maybe fix the fen issues
        # board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

        #show the board 
        cv2.imshow("board",board)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return board

#analyze board position
def analyze_board(board):
    print("board: ",type(board))
    #board = Image.fromarray(np.uint8(board))
    print("board shape: ",board.shape)
    #board.save(img=board)
    print(get_fen_from_image(board))
    #current error: AttributeError: module 'PIL.Image' has no attribute 'resize'

print('working')
screen = screenshot()
print('screen shotted')
board = get_board(screen)
print('board processed')
fen = analyze_board(board)
print('board analized')