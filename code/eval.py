import numpy as np

import glob
import argparse

def cal_iiou(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    

    Returns
    -------
    float
        DESCRIPTION.
            iou has the value between 0.0 and 1.0

    """
    
    irrect1 = 0
    irrect2 = 0
    
    intersec = 0
                    
    return intersection / (box1_area + box2_area - intersection)

def eval_track(args: argparse.ArgumentParser) -> float:
    
    iou_sum = 0
    
    eval_img_list = glob.glob(args.eval_prefix)
    
    # Calculate iou in Evaluation Data
    for i, eval_img in enumerate(eval_img_list)
        
        point1 = None 
        point2 = None 
        
        iiou = cal_iiou(points1, points2)
        
        iou_sum += iou

    # Return average iou        
    return iou_sum/len(eval_img_list)