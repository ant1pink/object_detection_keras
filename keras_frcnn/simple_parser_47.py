import cv2
import os
import numpy as np
from glob import glob

def get_data(folder_name, class_name_path):
	found_bg = False
	folder_list =[ x for x in glob(folder_name+'*/') if 'no-logo' not in x]
	all_imgs = {}

	classes_count = {}

	class_mapping = {}

	class_name_dict ={}

	with open(class_name_path, 'r') as f:
		for line in f:
			line_split = line.strip().split('\t')
			(cname, cindex) = line_split
			class_name_dict[cindex] = cname

	print('Parsing annotation files')

	for input_path in folder_list:
		txt_list = [x for x in os.listdir(input_path) if '.txt' in x]
		for filename in txt_list:
			with open(os.path.join(input_path,filename),'r') as f:
				for line in f:
					line_split = line.strip().split(' ')
					(x1, y1, x2, y2, class_index, dummy_value, mask, difficult, truncated) = line_split
					class_name = class_name_dict[class_index]
					imagefile_name = os.path.join(input_path, filename.split('.')[0] + '.png')
					imagefile_name_s =  filename.split('.')[0]+'.png'
					if imagefile_name_s in os.listdir(input_path):
						if class_name not in classes_count:
							classes_count[class_name] = 1
						else:
							classes_count[class_name] += 1

						if class_name not in class_mapping:
							if class_name == 'bg' and found_bg == False:
								print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
								found_bg = True
							class_mapping[class_name] = len(class_mapping)

						if imagefile_name not in all_imgs:
							all_imgs[imagefile_name] = {}

							img = cv2.imread(imagefile_name)
							(rows,cols) = img.shape[:2]
							all_imgs[imagefile_name]['filepath'] = imagefile_name
							all_imgs[imagefile_name]['width'] = cols
							all_imgs[imagefile_name]['height'] = rows
							all_imgs[imagefile_name]['bboxes'] = []
							if np.random.randint(0, 6) > 0:
								all_imgs[imagefile_name]['imageset'] = 'train'
							else:
								all_imgs[imagefile_name]['imageset'] = 'test'

						all_imgs[imagefile_name]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

	print(len(all_imgs))
	all_data = []
	for key in all_imgs:
		all_data.append(all_imgs[key])
	print(len(all_data))
	# make sure the bg class is last in the list
	if found_bg:
		if class_mapping['bg'] != len(class_mapping) - 1:
			key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
			val_to_switch = class_mapping['bg']
			class_mapping['bg'] = len(class_mapping) - 1
			class_mapping[key_to_switch] = val_to_switch

	return all_data, classes_count, class_mapping



if __name__ == '__main__':
	folder_name = 'C:/Users/ruili2.LL/Desktop/flickr/FlickrLogos_47/train/'


	get_data(folder_name)
