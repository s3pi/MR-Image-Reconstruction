import os
import numpy as np
import nibabel as nib
import cv2 

def mkdir(path):
	try:
	    os.mkdir(path)
	except OSError:
		# print ("Creation of the directory %s failed" % path)
		pass

def min_max_normalization(img):
	#Volume normaization of each image, x_grad and y_grad.
	img_vol_min = np.min(img)
	img_vol_max = np.max(img)
	img = (img - img_vol_min) / (img_vol_max - img_vol_min) #img.shape == (img_row, slices, img_col : 259, 51, 259)
	
	return img

def find_x_and_y_grad(img, resizeTo):
	x_grad = []
	y_grad = []
	for k in range(len(img)):	
		x_grad.append(cv2.Sobel(np.reshape(img[k,:,:],[resizeTo,resizeTo]),cv2.CV_64F,1,0,ksize=5))
		y_grad.append(cv2.Sobel(np.reshape(img[k,:,:],[resizeTo,resizeTo]),cv2.CV_64F,0,1,ksize=5))
	x_grad=np.asarray(x_grad)
	y_grad=np.asarray(y_grad)

	return x_grad, y_grad

def treshold(img, img_type):
	if img_type == "img":
		z,x,y=np.where(img[:,:,:] < 0.1)
		img[z,x,y]=0
	elif img_type == "grad":
		z,x,y=np.where(img[:,:,:] < 0.001)	
		img[z,x,y]=0

	return img

def load_data_slices_into_folders():
	i_s = 336
	z=range(31, 231)
	num_of_slices = len(list(range(31, 231)))
	resizeTo = i_s + 3 	# Why Resize by 3? 
	data_path = "/home/ada/Preethi/Neural_Based_MRI/T1_T2/MICCAI/Data/TrainingData/"
	pro_data_path = "/home/ada/Preethi/Neural_Based_MRI/T1_T2/MICCAI/Code/processed_data/"
	for each_data_type in ['T1_Brain_Mask', 'T2_Brain_Mask', 'T1', 'T2', 'T2_down_4', 'T2_down_8', 'T2_down_16']:
		each_pro_data_type_path = os.path.join(pro_data_path, each_data_type)
		mkdir(each_pro_data_type_path)
		data_type_path = os.path.join(data_path, each_data_type)
		all_subjects = sorted(os.listdir(data_type_path))
		for each_subject in all_subjects:
			temp = np.zeros([num_of_slices, resizeTo, resizeTo])
			img = nib.load(data_type_path + '/' + each_subject)
			affine = img.affine
			img = img.get_data()
			temp[:,3:,3:] = img[z, :, :]
			img = temp
			img_norm = min_max_normalization(img)
			img_norm = treshold(img_norm, "img")
			
			if (each_data_type != 'T1_Brain_Mask') and (each_data_type != 'T2_Brain_Mask'):
				x_grad, y_grad = find_x_and_y_grad(img, resizeTo)
				x_grad_norm = min_max_normalization(x_grad)
				x_grad_norm = treshold(x_grad_norm, "grad")
				y_grad_norm = min_max_normalization(y_grad)
				y_grad_norm = treshold(y_grad_norm, "grad")

			each_subject = each_subject.split('.')[0]
			each_subject_path = os.path.join(each_pro_data_type_path, each_subject)
			mkdir(each_subject_path)
			i = 0
			for slice_num in z:
				each_slice_path = os.path.join(each_subject_path, 'slice_num_' + ('0'* (3 - len(str(slice_num)))) + str(slice_num))
				mkdir(each_slice_path)
				each_slice_img_path = os.path.join(each_slice_path, "Img")
				mkdir(each_slice_img_path)
				np.save(each_slice_img_path + '/Img.npy', img_norm[i])
				if (each_data_type != 'T1_Brain_Mask') and (each_data_type != 'T2_Brain_Mask'):
					each_slice_x_grad_path = os.path.join(each_slice_path, "X_grad")
					mkdir(each_slice_x_grad_path)
					np.save(each_slice_x_grad_path + '/x_grad.npy', x_grad_norm[i])
					each_slice_y_grad_path = os.path.join(each_slice_path, "y_grad")
					mkdir(each_slice_y_grad_path)
					np.save(each_slice_y_grad_path + '/y_grad.npy', y_grad_norm[i])
				i += 1


load_data_slices_into_folders()

	