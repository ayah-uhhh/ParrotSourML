# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 19:31:59 2023

@author: ayaha
"""
import cv2
"""
 Labels
 1: Azimuth, 2: Range, 3: Wall
 4: Ladder,  5: Champagne 6: Vic
 7: Single
 """
class ShapeDetector:
 def __init__(self):
     pass
 def detect(self, c):
     shape = "unidentified"
     peri = cv2.arcLength(c, True) 
     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
     
     """ 
     n is picture number, 
     i is cluster of air craft, 
     j is the individual aircraft
     
     if there is only one cluster --> SINGLE
     
     if there are 2 clusters with a greater distance in the 
         y axis than the x axis between them --> AZIMUTH
        
     if there are 2 clusters with a greater distance in the 
        x axis than the y axis between them --> RANGE
        
     if there are more than 2 clusters with a greater distance in the 
        y axis than the x axis between them --> WALL
    
     if there are more than 2 clusters with a greater distance in the 
        x axis than the y axis between them --> LADDER
     
     if there are 3 clusters, where 2 clusters have a minimal difference between
         their x coordinates, and 1 cluster has a larger x coordinate distance 
         and smaller X1 value compated to the other 2 clusters --> CHAMPAGNE
         
     if there are 3 clusters, where 2 clusters have a minimal difference between
        their x coordinates, and 1 cluster has a larger x coordinate distance 
        and smaller X1 value compated to the other 2 clusters --> VIC
    """
    
    
def get_cluster_type(n, i, j, flat):
    # Single cluster
    if len(flat[f"{n}_groups"]) == 1:
        return "SINGLE"
    
    # Two clusters
    elif len(flat[f"{n}_groups"]) == 2:
        x1_min1 = min(flat[f"{n}_groups"][0], key=lambda k: flat[f"{n}_groups"][0][k][i][j]["startPos_x"])
        x1_min2 = min(flat[f"{n}_groups"][1], key=lambda k: flat[f"{n}_groups"][1][k][i][j]["startPos_x"])
        x1_diff = abs(flat[f"{n}_groups"][0][x1_min1][i][j]["startPos_x"] - flat[f"{n}_groups"][1][x1_min2][i][j]["startPos_x"])
        y_diff = abs(flat[f"{n}_groups"][0][x1_min1][i][j]["startPos_y"] - flat[f"{n}_groups"][1][x1_min2][i][j]["startPos_y"])
        if y_diff > x1_diff:
            return "AZIMUTH"
        else:
            return "RANGE"
        
    # Multiple clusters
    else:
        x1_vals = [flat[f"{n}_groups"][k][min(flat[f"{n}_groups"][k], key=lambda j: flat[f"{n}_groups"][k][j][i][j]["startPos_x"])][i][j]["startPos_x"] for k in range(len(flat[f"{n}_groups"]) )]
        x1_diffs = [abs(x1_vals[k] - x1_vals[l]) for k in range(len(x1_vals)-1) for l in range(k+1, len(x1_vals))]
        y_diffs = [abs(flat[f"{n}_groups"][k][min(flat[f"{n}_groups"][k], key=lambda j: flat[f"{n}_groups"][k][j][i][j]["startPos_x"])][i][j]["startPos_y"] - flat[f"{n}_groups"][l][min(flat[f"{n}_groups"][l], key=lambda j: flat[f"{n}_groups"][l][j][i][j]["startPos_x"])][i][j]["startPos_y"]) for k in range(len(flat[f"{n}_groups"]) - 1) for l in range(k+1, len(flat[f"{n}_groups"]))]
        
        if max(y_diffs) > max(x1_diffs):
            return "WALL"
        elif max(y_diffs) < max(x1_diffs):
            return "LADDER"
        else:
            x1_sorted = sorted(x1_vals)
            if x1_sorted[1] - x1_sorted[0] <= 50 and x1_sorted[2] - x1_sorted[1] <= 50 and x1_sorted[2] - x1_sorted[0] > 150:
                return "CHAMPAGNE"
            elif x1_sorted[1] - x1_sorted[0] <= 50 and x1_sorted[2] - x1_sorted[1] <= 50 and x1_sorted[2] - x1_sorted[0] <= 150:
                return "VIC"