import numpy as np
import cv2
import glob
import os
from shapely.geometry import Polygon
import ast
import json

import numpy as np
import sys
from utils import calculate_polygon_iou_precision_recall_f1, calculate_polygon_iou_matrix
from utils import calculate_bbox_iou_precision_recall_f1, calculate_bbox_iou_matrix
from utils import fix_self_intersection, draw_polygons, draw_multiple_polygons

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

if __name__ =='__main__':
	prediction_results = sys.argv[1] #'../Results/Polygon/Ours/docExtractor_v2/polygonbb_results_7_5_3_latest/'
	print(prediction_results)
	out_visual_folder = sys.argv[2]

	gt_results = './Groundtruth/Manuscripts_GT_Polygons/'
	list_file = open('list_of_file_use.txt','r').readlines()
	list_file = [s.rstrip('\n') for s in list_file]
	list_file_not_use = open('list_used_file_for_finetuning_docExtractor.txt','r').readlines()
	list_iou = []
	list_p  =[]
	list_r =[]
	list_f1 =[]
	list_ap50 =[]
	list_ap75 = []
	img_dir ='./uptodate-manuscripts/'
	debug = False
	out_dir =f'./Visual/Polygons/{out_visual_folder}'
	our_dir_txt =f'./Text/'
	os.makedirs(out_dir,exist_ok = True)
	os.makedirs(our_dir_txt,exist_ok = True)
	our_dir_txt_file = open(os.path.join(our_dir_txt,out_visual_folder+'.txt'),'w')
	our_dir_txt_file.write('Filename'+'\t'+'Iou'+'\t'+'Precision'+'\t'+'Recall'+'\t'+'F1'+'\t'+'AP50'+'\t'+'AP75'+'\n')
	if len(sys.argv)>3:
		debug=True
	for idx, file in enumerate(sorted(list_file)):
		if file in list_file_not_use:
			continue
		data={}
		base_name = os.path.basename(file)
		#print(idx, base_name)
		if not os.path.exists(os.path.join(prediction_results,file)):
			continue
		#list_bbox_predict = open(os.path.join(prediction_results,file),'r').readlines()
		list_bbox_gt = open(os.path.join(gt_results,file),'r').readlines()
		predicted_polygons = []
		ground_truth_polygons = []
		#if debug:
		img = cv2.imread(os.path.join(img_dir,base_name[0:-4]+'.jpg'))
		h_ori, w_ori = img.shape[:2]

		size = 200
		with open(os.path.join(prediction_results,file),'r') as p_file:
			for line in p_file:
				if line.startswith("Contour"):
					points_str = line.split("((")[1].split("))")[0]
					point_list = [tuple(map(int, point.split())) for point in points_str.split(", ")]

					try:
						p_polygon = Polygon(point_list)
						if p_polygon.area > size or not p_polygon.is_empty:
							p_polygon = fix_self_intersection(p_polygon)
							if p_polygon== None:
								continue
							predicted_polygons.append(p_polygon)
					except:
						continue

		for bbox in list_bbox_gt:
			bbox = bbox.rstrip('\n').split('\t')

			point_list = [ast.literal_eval(t) for t in bbox]
			polygon = Polygon(point_list)
			polygon = fix_self_intersection(polygon)

			ground_truth_polygons.append(polygon)

		if len(predicted_polygons)==0:
			print(base_name,'is empty')
			list_iou.append(0)
			list_p.append(0)
			list_r.append(0)
			list_f1.append(0)
			list_ap50.append(0)
			list_ap75.append(0)
			our_dir_txt_file.write(base_name+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\n')
			continue
		iou_matrix  = calculate_polygon_iou_matrix(ground_truth_polygons, predicted_polygons)
		col_inds = np.argmax(iou_matrix, axis=1)

		row_inds = np.arange(iou_matrix.shape[0])
		tmp_1 = []
		tmp_2 = []
		tmp_3 = []
		tmp_4 = []
		tmp_5 = []
		tmp_6 = []
		for row, col in zip(row_inds,col_inds):
			i,p,r,f1,ap_50, ap_75 = calculate_polygon_iou_precision_recall_f1(ground_truth_polygons[row],predicted_polygons[col])
			#print(i,p,r,f1,ap_50, ap_75)
			tmp_1.append(i)
			tmp_2.append(p)
			tmp_3.append(r)
			tmp_4.append(f1)
			tmp_5.append(ap_50)
			tmp_6.append(ap_75)
			if debug:
				# if ground_truth_polygons[row].type =='MultiPolygon':
				# 	draw_multiple_polygons(img, ground_truth_polygons[row])
				# 	for polygon in ground_truth_polygons[row].geoms:
				# 		if polygon.exterior is not None and polygon.exterior.coords:
				# 			pos = polygon.exterior.coords[0]
				# else:
				# 	draw_polygons(img, ground_truth_polygons[row])
				# 	pos = list(ground_truth_polygons[row].exterior.coords)[0]

				if predicted_polygons[col].type =='MultiPolygon':
					draw_multiple_polygons(img, predicted_polygons[col],(0,255,0))
				else:
					draw_polygons(img,predicted_polygons[col],(0,255,0),5)
				
				desp_txt = "{} -> {}, iou: {:.2f}".format(row,col,iou_matrix[row,col])
				
				#cv2.putText(img, desp_txt, (int(pos[0]),int(pos[1])),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

				cv2.imwrite(os.path.join(out_dir,base_name[0:-4]+'.jpg'),img)
		avg_iou = sum(tmp_1)/max(len(tmp_1),1e-6)
		list_iou.append(avg_iou)
		avg_p = sum(tmp_2)/max(len(tmp_2),1e-6)
		list_p.append(avg_p)
		avg_r = sum(tmp_3)/max(len(tmp_3),1e-6)
		list_r.append(avg_r)
		avg_f1 = sum(tmp_4)/max(len(tmp_4),1e-6)
		list_f1.append(avg_f1)
		avg_ap50 = sum(tmp_5)/max(len(tmp_5),1e-6)
		list_ap50.append(avg_ap50)
		avg_ap75 = sum(tmp_6)/max(len(tmp_6),1e-6)
		list_ap75.append(avg_ap75)
		our_dir_txt_file.write(f"{base_name}\t{avg_iou:.3f}\t{avg_p:.3f}\t{avg_r:.3f}\t{avg_f1:.3f}\t{avg_ap50:.3f}\t{avg_ap75:.3f}\n")

	print(sum(list_iou)/len(list_iou))
	print(sum(list_p)/len(list_p))
	print(sum(list_r)/len(list_r))
	print(sum(list_f1)/len(list_f1))
	print(sum(list_ap50)/len(list_ap50))
	print(sum(list_ap75)/len(list_ap75))
	our_dir_txt_file.close()