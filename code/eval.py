import cv2
import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

from roi import roi
from utils import point_sort, build_parser

import glob
import argparse

def cal_iarea(*_p: np.ndarray) -> list:
    
    area = list()
    
    for p in _p:    
        a = p[0] - (p[1] + p[3])/2
        b = p[2] - (p[1] + p[3])/2
        
        s = np.abs(a[0]*b[1] - a[1]*b[0])
        
        area.append(s)
    
    return area

def cal_ipnt(p1: np.ndarray, p2: np.ndarray):
    
    # Set Bary Center
    bary_cnt1 = np.mean(p1, axis=0)
    bary_cnt2 = np.mean(p2, axis=0)
    global itsec_p
    global cnt
    cnt = (bary_cnt1+bary_cnt2)/2
    
    # Generate Intersection Points
    itsec_p = list()
    for i, _a in enumerate(p1):
        _b = p1[-len(p1) + i + 1]
        
        for j, _c in enumerate(p2):
            _d = p2[-len(p2) + j + 1]
            
            # Intersection Point
            
            A1 = (_a*_b[::-1])[0] - (_a*_b[::-1])[1]
            A2 = (_c*_d[::-1])[0] - (_c*_d[::-1])[1]
            
            D = (_a-_b)*((_c-_d)[::-1])
            #print(D)
            D = D[0] - D[1]
            
            _p_x = (A1*(_c[0]-_d[0])-A2*(_a[0]-_b[0]))/D
            _p_y = (A1*(_c[1]-_d[1])-A2*(_a[1]-_b[1]))/D
        
            itsec_p.append(np.array([_p_x, _p_y]))
        
    itsec_p = np.array(itsec_p)
    itsec_p = np.concatenate((itsec_p, p1, p2))    
    
    # Exception inf, NaN
    np.nan_to_num(itsec_p, copy=False)
    itsec_p = np.array(itsec_p ,dtype=np.int32)
    
    # None Maximum Suprression 
    i=0
    while i < len(itsec_p):
        _brk = False
        j=0
        while j < len(itsec_p):
            _temp = np.linalg.norm((itsec_p[i]-itsec_p[j]))
            if _temp < 50 and i != j:
                _a = np.linalg.norm(np.abs(itsec_p[i])-cnt)
                _b = np.linalg.norm(np.abs(itsec_p[j])-cnt)

                _del_i = j if _a <= _b else i
                itsec_p = np.delete(itsec_p, _del_i, axis=0)
                
                _brk = True
                break
            j+=1
        
        if _brk != True: i+=1
                
    # Extract Close Points from Bary Center
    itsec_p_idx = np.argsort(np.linalg.norm(np.abs(itsec_p)-cnt, axis=1))
   
    return itsec_p[itsec_p_idx][:4]

def cal_iiou(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    

    Returns
    -------
    float
        DESCRIPTION.
            iou has the value between 0.0 and 1.0

    """
    
    # Point Order
    # [x, y] = [row, column]
    # [top-left, bottom-left, bottom-right, top-right]
    p1, p2 = point_sort(p1), point_sort(p2)
    p3 = point_sort(cal_ipnt(p1, p2))
    
    box1_area, box2_area, box_inter = cal_iarea(p1, p2, p3)
        
    if box_inter > box1_area + box2_area:
        #print(p1, p2, p3)
        #print()
        #print(box1_area, box2_area, box_inter)
        #raise ValueError("[Exception] box_inter is better than Union")
        return None
                
    return box_inter / (box1_area + box2_area - box_inter)

def arima_fit(*data_arg, p, d, q) -> float:
    
    fit = list()
    
    for data in data_arg:
        model = sm.tsa.arima.ARIMA(data, order=(p, d, q))
        model_fit = model.fit()
        
        fit.append(float(model_fit.forecast()))
    
    return fit

def eval_track(gt_box, p, d, q) -> float:
    
    iou_sum = 0
    iou_num = 0
    
    t_idx = 30
    
    p1_x, p1_y = gt_box[:,0,0], gt_box[:,0,1]
    p2_x, p2_y = gt_box[:,1,0], gt_box[:,1,1]
    p3_x, p3_y = gt_box[:,2,0], gt_box[:,2,1]
    p4_x, p4_y = gt_box[:,3,0], gt_box[:,3,1]
    
    # Calculate iou in Evaluation Data
    for i in range(len(gt_box)):
        if t_idx+i >= len(gt_box)-1:
            print("break")
            break
        
        p1_x_res, p1_y_res, p2_x_res, p2_y_res, \
            p3_x_res, p3_y_res, p4_x_res, p4_y_res = \
                arima_fit(p1_x[:t_idx+i],
                          p1_y[:t_idx+i],
                          p2_x[:t_idx+i],
                          p2_y[:t_idx+i],
                          p3_x[:t_idx+i],
                          p3_y[:t_idx+i],
                          p4_x[:t_idx+i],
                          p4_y[:t_idx+i],
                          p, d, q)
         
        point1 = np.array([[int(p1_x_res), int(p1_y_res)],
                           [int(p2_x_res), int(p2_y_res)],
                           [int(p3_x_res), int(p3_y_res)],
                           [int(p4_x_res), int(p4_y_res)]])
        
        point2 = gt_box[t_idx+1+i] 
        
        iiou = cal_iiou(point1, point2)
        
        if iiou != None and 0.0 < iiou <= 1.0:
            iou_num += 1
            iou_sum += iiou
            #print("iiou :", iiou)
        else:
            #print("[Exception] iiou")
            continue

    # Return average iou
    return iou_sum/iou_num

if __name__ == "__main__":
    
    parser = build_parser()
    args = parser.pars_args()
    
    # Ground Truth Data
    gt_box = list()
    for i in glob.glob(args.eval_prefix):
        approxs, conts = roi(cv2.imread(i))
        gt_box.append(np.squeeze(approxs)[:,::-1])
    
    gt_box = np.array(gt_box)
        
    p = args.p
    d = args.d
    q = args.q
    
    # Evaluate
    eval_iiou = eval_track(gt_box, p, d, q)
    print(eval_iiou)

    #%timeit arima_fit
