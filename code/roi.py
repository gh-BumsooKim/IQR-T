import cv2
import numpy as np

# 3.43 ms ± 65.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#   when img.shape is (1280, 720, 3), s is 1
def roi(img: np.ndarray, s: int = 1) -> tuple:
    
    img1 = img
    
    h, w, _ = img.shape
    
    img2 = cv2.resize(img, (int(w/s), int(h/s)))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Pre-Processing
    ret, img2 = cv2.threshold(img2, 1, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3))
    img2 = cv2.erode(img2, kernel, iterations=3)
    img2 = cv2.dilate(img2, kernel, iterations=3)
    
    # Contour Detection
    conts, hry = cv2.findContours(img2,
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    img2 = np.stack([img2, img2, img2], axis=2)
    
    
    approxs = list()
    for cont in conts:
        approxs.append(cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True))
    
    #cv2.polylines(img2, [approx], True, (0, 0, 255), 6)
    
    if len(approxs[0]) != 4:
        #print(f"{approx}, Exception")
        return 0
    
    #cv2.imshow("test"+str(s), img2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return approxs, conts


if __name__ == "__main__":
    img = cv2.imread("test.png")
    
    global cont
    cont = roi(img, 8)
