import numpy as np

def point_sort(pts):
    index = np.argsort(pts[:, 0])

    pts_sorted = np.empty_like(pts)
    for i in range(len(pts)):
        pts_sorted[i] = pts[index[i]]
    
    tl = bl = br = tr = None
    if pts_sorted[0][1] < pts_sorted[1][1]:
        tl, tr = pts_sorted[0], pts_sorted[1]
    else:
        tr, tl = pts_sorted[0], pts_sorted[1]        
        
    if pts_sorted[2][1] < pts_sorted[3][1]:
        bl, br = pts_sorted[2], pts_sorted[3]
    else:
        br, bl = pts_sorted[2], pts_sorted[3]  

    pts = np.float32([tl, bl, br, tr])
    
    return pts