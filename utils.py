import numpy as np
import cv2
import glob
import os
from shapely.geometry import Polygon
import ast
import json
import numpy as np
from shapely.ops import unary_union

def calculate_iou(box1, box2):
    # Function to calculate Intersection over Union (IoU) between two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersect_x = max(x1, x2)
    intersect_y = max(y1, y2)
    intersect_w = max(0, min(x1 + w1, x2 + w2) - intersect_x)
    intersect_h = max(0, min(y1 + h1, y2 + h2) - intersect_y)

    area_intersect = intersect_w * intersect_h
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    iou = area_intersect / float(area_box1 + area_box2 - area_intersect)
    return iou

def calculate_bbox_iou_matrix(ground_truth_boxes, predicted_boxes):
    num_gt_boxes = len(ground_truth_boxes)
    num_pred_boxes = len(predicted_boxes)

    iou_matrix = np.zeros((num_gt_boxes, num_pred_boxes))

    for i in range(num_gt_boxes):
        for j in range(num_pred_boxes):
            iou_matrix[i, j] = calculate_iou(ground_truth_boxes[i], predicted_boxes[j])

    return iou_matrix


def calculate_bbox_iou_precision_recall_f1(box1, box2):
    """
    box1: gt
    box2: prediction
    """
    # Function to calculate Intersection over Union (IoU) between two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersect_x = max(x1, x2)
    intersect_y = max(y1, y2)
    intersect_w = max(0, min(x1 + w1, x2 + w2) - intersect_x)
    intersect_h = max(0, min(y1 + h1, y2 + h2) - intersect_y)

    area_intersect = intersect_w * intersect_h
    area_box1 = w1 * h1 +1e-6
    area_box2 = w2 * h2 +1e-6

    iou = area_intersect / float(area_box1 + area_box2 - area_intersect)
    
    precision = round(area_intersect/area_box2,3)
    recall = round(area_intersect/area_box1,3)
    f1 = round(2 *area_intersect / float(area_box1+area_box2+1e-6),3)
    ap_50 = 1 if iou>0.5 else 0
    ap_75 = 1 if iou>0.75 else 0
    return iou, precision, recall, f1, ap_50, ap_75




def calculate_polygon_iou(polygon1,polygon2):
    intersection_area = polygon1.intersection(polygon2).area

    # Calculate the area of union
    union_area = polygon1.union(polygon2).area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def calculate_polygon_iou_matrix(ground_truth_polygons, predicted_polygons):
    num_gt_polygons = len(ground_truth_polygons)
    num_pred_polygons = len(predicted_polygons)

    poylgon_iou_matrix = np.zeros((num_gt_polygons, num_pred_polygons))

    for i in range(num_gt_polygons):
        for j in range(num_pred_polygons):
            #print(i,j)
            #print(predicted_polygons[j])
            #print(predicted_polygons[j].area)
            poylgon_iou_matrix[i, j] = calculate_polygon_iou(ground_truth_polygons[i], predicted_polygons[j])

    return poylgon_iou_matrix

def calculate_polygon_iou_precision_recall_f1(polygon1, polygon2):
    """
    box1: gt
    box2: prediction
    """
    # Function to calculate Intersection over Union (IoU) between two bounding boxes
    intersection_area = polygon1.intersection(polygon2).area

    # Calculate the area of union
    union_area = polygon1.union(polygon2).area

    # Calculate IoU
    iou = intersection_area / union_area
    
    precision = round(intersection_area/polygon2.area,3)
    recall = round(intersection_area/polygon1.area,3)
    f1 = round(2 *intersection_area / float(polygon2.area+polygon1.area+1e-6),3)
    ap_50 = 1 if iou>0.5 else 0
    ap_75 = 1 if iou>0.75 else 0
    return iou, precision, recall, f1, ap_50, ap_75



def draw_polygons(image, polygon, color=(255, 0, 0), thickness=2):
        # Extract the exterior coordinates of the polygon
        exterior_coords = list(polygon.exterior.coords)

        # Convert the coordinates to a numpy array
        vertices = np.array(exterior_coords, np.int32)
        vertices = vertices.reshape((-1, 1, 2))

        # Draw the polygon on the image
        cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=thickness)
def draw_multiple_polygons(image, polygons,color=(255, 0, 0), thickness=2):
    for polygon in polygons.geoms:
        draw_polygons(image, polygon, color, thickness=2)

        
def fix_self_intersection(polygon):
    # Buffer the polygon with a distance of 0 to attempt to fix self-intersections
    buffered_polygon = polygon.buffer(0)

    # If the buffer results in a MultiPolygon, try to merge it into a single Polygon
    if buffered_polygon.is_empty:
        return None
    elif buffered_polygon.is_simple:
        return buffered_polygon
    elif buffered_polygon.is_multigeometry:
        return unary_union(buffered_polygon)

def resize_polygon(polygons, scale, h , w):
    x_points = [int(element[0] * scale) for element in polygons]
    y_points = [int(element[1] * scale) for element in polygons]

    x_points = [int(element) if element < w else w for element in x_points]
    y_points = [int(element) if element < h else h for element in y_points]
    x_points = [int(element) if element > 0 else 0 for element in x_points]
    y_points = [int(element) if element > 0 else 0 for element in y_points]

    assert max(x_points) <= w
    assert min(x_points) >= 0
    assert max(y_points) <= h
    assert min(y_points) >= 0
    resized_polygons = list(zip(x_points, y_points))
    return resized_polygons