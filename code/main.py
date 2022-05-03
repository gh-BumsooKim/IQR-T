import cv2
import numpy as np

from roi import roi
from utils import point_sort

if __name__ == "__main__":
    
    # Contnet
    img = cv2.imread("test2.png")
    
    content = list()    
    content.append(cv2.imread("./human_01.png"))
    content.append(cv2.imread("./human_03.png"))
    content.append(cv2.imread("./human_02.png"))
    
    s = 1
    
    global approx
    global conts
    approxs, conts = roi(img, s)

    global cont_results
    cont_results = list()
    for i, approx in enumerate(approxs):
        h, w, _ = img.shape
        h_c, w_c, _ = content[i].shape
    
        pts1 = np.float32([[0,0],[w_c,0],[w_c, h_c],[0, h_c]])
        pts2 = np.float32([approx[0][0], approx[3][0], approx[2][0], approx[1][0]])
    
        # Sort
        pts2 = point_sort(pts2)
    
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        cont_result = cv2.warpPerspective(content[i], mtrx, (int(w/s), int(h/s)))
        cont_results.append(cont_result)
    
    img_result = np.uint8(np.sum(cont_results, axis=0))
    
    ## 
    #cv2.imshow("Test2", img)
    cv2.imshow("Test", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("result.png", img_result)
    
    
    