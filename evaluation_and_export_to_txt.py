import cv2
import glob
import os
import sys
import ast
import json
import numpy as np
from shapely.geometry import Polygon
from utils import (
    calculate_polygon_iou_precision_recall_f1, 
    calculate_polygon_iou_matrix, 
    fix_self_intersection, 
    draw_polygons, 
    draw_multiple_polygons, 
    resize_polygon
)

def process_file(predictions_path, gt_path, file, img_dir, out_dir, debug=False):
    predicted_polygons = []
    ground_truth_polygons = []

    img = cv2.imread(os.path.join(img_dir, os.path.splitext(file)[0] + '.jpg'))
    if img is None:
        return None

    h_ori, w_ori = img.shape[:2]
    prediction_file = os.path.join(predictions_path, file)
    gt_file = os.path.join(gt_path, file)

    if not os.path.exists(prediction_file):
        return None

    with open(prediction_file, 'r') as p_file:
        for line in p_file:
            if line.startswith("Contour"):
                points_str = line.split("((")[1].split("))")[0]
                point_list = [tuple(map(int, point.split())) for point in points_str.split(", ")]
                if len(sys.argv) > 3:
                    point_list = resize_polygon(point_list, 4, h_ori, w_ori)

                try:
                    polygon = Polygon(point_list)
                    if polygon.area > 50:
                        polygon = fix_self_intersection(polygon)
                        if polygon:
                            predicted_polygons.append(polygon)
                except:
                    continue

    with open(gt_file, 'r') as gt_file:
        for bbox in gt_file:
            point_list = [ast.literal_eval(point) for point in bbox.rstrip('\n').split('\t')]
            polygon = fix_self_intersection(Polygon(point_list))
            if polygon:
                ground_truth_polygons.append(polygon)

    return img, predicted_polygons, ground_truth_polygons

def main():
    predictions_path = sys.argv[1]
    out_visual_folder = sys.argv[2]
    gt_results = './Groundtruth/Manuscripts_GT_Polygons/'
    img_dir = './uptodate-manuscripts/'
    out_dir = f'./Visual/Polygons/{out_visual_folder}'
    out_text_dir = f'./Text/{out_visual_folder}.txt'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_text_dir), exist_ok=True)

    with open(out_text_dir, 'w') as result_file:
        result_file.write('Filename\tIoU\tPrecision\tRecall\tF1\tAP50\tAP75\n')
        
        list_file = [line.strip() for line in open('list_of_file_use.txt', 'r').readlines()]
        excluded_files = set(line.strip() for line in open('list_used_file_for_finetuning_docExtractor.txt', 'r').readlines())

        metrics = {'iou': [], 'precision': [], 'recall': [], 'f1': [], 'ap50': [], 'ap75': []}

        for idx, file in enumerate(sorted(list_file)):
            if idx == 10 or file in excluded_files:
                break

            result = process_file(predictions_path, gt_results, file, img_dir, out_dir, debug=(len(sys.argv) > 4))
            if not result:
                result_file.write(f"{file}\t0\t0\t0\t0\t0\t0\n")
                continue

            img, predicted_polygons, ground_truth_polygons = result

            if not predicted_polygons:
                result_file.write(f"{file}\t0\t0\t0\t0\t0\t0\n")
                continue

            iou_matrix = calculate_polygon_iou_matrix(ground_truth_polygons, predicted_polygons)
            col_inds = np.argmax(iou_matrix, axis=1)

            scores = {'iou': [], 'precision': [], 'recall': [], 'f1': [], 'ap50': [], 'ap75': []}
            for row, col in zip(range(len(iou_matrix)), col_inds):
                iou, precision, recall, f1, ap50, ap75 = calculate_polygon_iou_precision_recall_f1(
                    ground_truth_polygons[row], predicted_polygons[col]
                )
                scores['iou'].append(iou)
                scores['precision'].append(precision)
                scores['recall'].append(recall)
                scores['f1'].append(f1)
                scores['ap50'].append(ap50)
                scores['ap75'].append(ap75)

                if len(sys.argv) > 4:
                    draw_polygons(img, ground_truth_polygons[row])
                    draw_polygons(img, predicted_polygons[col], color=(0, 255, 0), thickness=5)

            avg_scores = {key: sum(values) / max(len(values), 1e-6) for key, values in scores.items()}
            metrics['iou'].append(avg_scores['iou'])
            metrics['precision'].append(avg_scores['precision'])
            metrics['recall'].append(avg_scores['recall'])
            metrics['f1'].append(avg_scores['f1'])
            metrics['ap50'].append(avg_scores['ap50'])
            metrics['ap75'].append(avg_scores['ap75'])

            result_file.write(f"{file}\t{avg_scores['iou']:.3f}\t{avg_scores['precision']:.3f}\t"
                              f"{avg_scores['recall']:.3f}\t{avg_scores['f1']:.3f}\t"
                              f"{avg_scores['ap50']:.3f}\t{avg_scores['ap75']:.3f}\n")
            cv2.imwrite(os.path.join(out_dir, os.path.splitext(file)[0] + '.jpg'), img)

    # Print averages
    for metric_name, values in metrics.items():
        print(f"{metric_name.capitalize()}: {sum(values) / len(values):.3f}")

if __name__ == '__main__':
    main()
