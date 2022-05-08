import cv2
import numpy as np

#import open3d

import glob
import time
import screeninfo as sinfo

from roi import roi
from utils import point_sort, build_parser
#from dcpct import calibration

if __name__ == "__main__":
    # argparser
    parser = build_parser()
    args = parser.parse_args()
    
    # Screen Information for Projection
    monitors = sinfo.get_monitors()
    projector = monitors[1]
    
    # 0-1. Peform Camera-Projector Calibration
    #calibration()
    
    # 0-2. Load Content Image
    content = [cv2.imread(i) for i in glob.glob(args.cont_prefix)]
    
    if len(content) == 0:
        print("[Exception] Content Image is not exist,\
              it is replaced to arbitary contents.")
        temp = np.ones((360, 360), dtype=np.uint8)
        content = [np.stack([temp, temp*255, temp], axis=2),
                   np.stack([temp*255, temp, temp], axis=2),
                   np.stack([temp, temp, temp*255], axis=2)]
    
    # 1. Input Depth Map from RGBD Camera
    
    print("[Success] 01. RGBD Camera Connection")
    img = cv2.imread("test2.png")
    
    # 2. ROI Extraction
    s_time = time.time()
    s = 1
    
    global approx
    global conts
    approxs, conts = roi(img, s)

    # 3. Region Tracking
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
    
    
    # 4. Content Image Mapping
    img_result = np.uint8(np.sum(cont_results, axis=0))
    
    # 5. Projection Mapping 
    # np.hstack()
        # A Window for Comparison
    #cv2.imshow("Test2", img)
        # B Window for Projection
    
    e_time = time.time()
    print(f"FPS : {round(1/(e_time-s_time),1)} \t {int((e_time-s_time)*1000)}ms", end='\r')
    
    cv2.imshow("Test", img_result)
    cv2.waitKey(0)
        
    cv2.destroyAllWindows()
    cv2.imwrite("result.png", img_result)
    
    print()
    print("[Success] 05. Program Termination")
    
    
