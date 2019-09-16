from models import encoder, decoder, autoencoder, hourglass, net2
from keras.layers import Input, Add
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
from contextlib import redirect_stdout
import os
import numpy as np
from sklearn.utils import shuffle
import math
import cv2
import nibabel as nib

def root_mean_sq_GxGy(y_t, y_p):
    print("in loss function")
    print(K.int_shape(y_p))
    where = tf.not_equal(y_t[:,:,:,3], 0)
    # print(type(where))
    a_t=tf.boolean_mask(y_t[:,:,:,:3],where,name='boolean_mask')#this is not necessary?
    a_p=tf.boolean_mask(y_p[:,:,:,:3],where,name='boolean_mask')
    return (K.sqrt(K.mean((K.square(a_t-a_p)))))

def make_model():
    # Making all the modules of the model architecture
    i_s = 336
    encoder_inp_shape = (i_s,i_s,3)
    enc = encoder(encoder_inp_shape)
    hg_inp_shape_1 = (i_s // 4, i_s // 4, 512)
    hg1 = hourglass(hg_inp_shape_1)
    hg_inp_shape_2 = (i_s // 4, i_s // 4, 256)
    hg2 = hourglass(hg_inp_shape_2)
    decoder_inp_shape = (i_s // 4, i_s // 4, 256)
    dec = decoder(decoder_inp_shape)
    proSR = net2(encoder_inp_shape)

    # Making the graph by connecting all the moduless of the model architecture
    # Each of this model can be seen as a layer now.
    input_tensor_1 = Input(encoder_inp_shape)
    input_tensor_2 = Input(encoder_inp_shape)
    part1 = enc(input_tensor_1)
    part2 = hg1(part1)
    part3 = hg2(part2)
    part4 = dec(part3)
    part5 = proSR(input_tensor_2)
    output = Add()([part4, part5])
    model = Model([input_tensor_1, input_tensor_2], output)
    model.compile(loss=root_mean_sq_GxGy, optimizer = RMSprop())

    with open('hourglass_sr_t1_t2.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    return model

def make_path_matrices(root_folder): 
    img_matrix = []
    x_grad_matrix = []
    y_grad_matrix = []
    all_subs = sorted(os.listdir(root_folder))
    for each_sub in all_subs:
        each_sub_imgs = []
        each_sub_x_grads = []
        each_sub_y_grads = []
        all_slices_path = os.path.join(root_folder, each_sub)
        all_slices = sorted(os.listdir(all_slices_path))
        for each_slice in all_slices:
            all_channels_path = os.path.join(all_slices_path, each_slice)
            all_channels = os.listdir(all_channels_path)
            for each_channel in all_channels:
                each_channel_path = os.path.join(all_channels_path, each_channel)
                files = os.listdir(each_channel_path)
                # Each folder has only one file.
                if files[0].startswith('Img'):
                    each_sub_imgs.append(os.path.join(each_channel_path, files[0]))
                elif files[0].startswith('x_grad'):
                    each_sub_x_grads.append(os.path.join(each_channel_path, files[0]))
                elif files[0].startswith('y_grad'):
                    each_sub_y_grads.append(os.path.join(each_channel_path, files[0]))
        img_matrix.append(each_sub_imgs)
        x_grad_matrix.append(each_sub_x_grads)
        y_grad_matrix.append(each_sub_y_grads)
    
    img_matrix = np.asarray(img_matrix)
    x_grad_matrix = np.asarray(x_grad_matrix)
    y_grad_matrix = np.asarray(y_grad_matrix)

    return img_matrix, x_grad_matrix, y_grad_matrix

def train_model(fold):
    train_T1 = "/home/ada/Preethi/Neural_Based_MRI/T1_T2/MICCAI/Code/processed_data/T1/"
    T1_img_path_matrix, T1_x_grad_path_matrix, T1_y_grad_path_matrix = make_path_matrices(train_T1)
    train_T2_downgraded = "/home/ada/Preethi/Neural_Based_MRI/T1_T2/MICCAI/Code/processed_data/T2_down_8"
    T2_down_img_path_matrix, T2_down_x_grad_path_matrix, T2_down_y_grad_path_matrix = make_path_matrices(train_T2_downgraded)
    train_T2 = "/home/ada/Preethi/Neural_Based_MRI/T1_T2/MICCAI/Code/processed_data/T2"
    T2_img_path_matrix, T2_x_grad_path_matrix, T2_y_grad_path_matrix = make_path_matrices(train_T2)
    T1_mask_path = "/home/ada/Preethi/Neural_Based_MRI/T1_T2/MICCAI/Code/processed_data/T1_Brain_Mask"
    T1_mask_img_path_matrix, dummy, dummy = make_path_matrices(T1_mask_path)
    T2_mask_path = "/home/ada/Preethi/Neural_Based_MRI/T1_T2/MICCAI/Code/processed_data/T2_Brain_Mask"
    T2_mask_img_path_matrix, dummy, dummy = make_path_matrices(T2_mask_path)
    resizeTo = 336 #original dimentions of the slice: 336 x 336 x 3
    num_of_subjects = 5
    num_of_slices = len(list(range(31, 231)))

    valid_T1_mask_img = []
    [valid_T1_mask_img.append(np.load(T1_mask_img_path_matrix[fold, i])[3:, 3:]) for i in range(num_of_slices)]
    valid_T1_mask_img = np.asarray(valid_T1_mask_img)

    valid_data_hg=np.zeros(shape=[num_of_slices,resizeTo, resizeTo, 3])
    valid_data_sr=np.zeros(shape=[num_of_slices,resizeTo, resizeTo, 3])
    valid_label=np.zeros(shape=[num_of_slices,resizeTo, resizeTo, 3])
    for k in range(num_of_slices):
        valid_data_hg[k,:,:,0] = np.load(T1_img_path_matrix[fold, k])[3:, 3:]
        valid_data_hg[k,:,:,1] = np.load(T1_x_grad_path_matrix[fold, k])[3:, 3:]
        valid_data_hg[k,:,:,2] = np.load(T1_y_grad_path_matrix[fold, k]) [3:, 3:]
        valid_data_sr[k,:,:,0] = np.load(T2_down_img_path_matrix[fold, k])[3:, 3:]
        valid_data_sr[k,:,:,1] = np.load(T2_down_x_grad_path_matrix[fold, k])[3:, 3:]
        valid_data_sr[k,:,:,2] = np.load(T2_down_y_grad_path_matrix[fold, k]) [3:, 3:]
        valid_label[k,:,:,0] = np.load(T2_img_path_matrix[fold, k])[3:, 3:]
        valid_label[k,:,:,1] = np.load(T2_x_grad_path_matrix[fold, k])[3:, 3:]
        valid_label[k,:,:,2] = np.load(T2_y_grad_path_matrix[fold, k]) [3:, 3:]
    #for optimized multiplication
    for k in range(3):
        valid_data_hg[:,:,:,k] = valid_T1_mask_img * valid_data_hg[:,:,:,k]
        valid_data_sr[:,:,:,k] = valid_T1_mask_img * valid_data_sr[:,:,:,k]
        valid_label[:,:,:,k] = valid_T1_mask_img * valid_label[:,:,:,k]

    train_T1_mask_img_paths = []
    train_data_hg_img_paths = []
    train_data_hg_x_grad_paths = []
    train_data_hg_y_grad_paths = []
    train_data_sr_img_paths = []
    train_data_sr_x_grad_paths = []
    train_data_sr_y_grad_paths = []
    train_label_img_paths = []
    train_label_x_grad_paths = []
    train_label_y_grad_paths = []
    
    l = 0
    for j in (list(range(fold)) + list(range(fold + 1, num_of_subjects))):
        for k in range(num_of_slices):
            train_T1_mask_img_paths.append(T1_mask_img_path_matrix[j, k])
            
            train_data_hg_img_paths.append(T1_img_path_matrix[j, k])
            train_data_hg_x_grad_paths.append(T1_x_grad_path_matrix[j, k])
            train_data_hg_y_grad_paths.append(T1_y_grad_path_matrix[j, k])

            train_data_sr_img_paths.append(T2_down_img_path_matrix[j, k])
            train_data_sr_x_grad_paths.append(T2_down_x_grad_path_matrix[j, k])
            train_data_sr_y_grad_paths.append(T2_down_y_grad_path_matrix[j, k])
            
            train_label_img_paths.append(T2_img_path_matrix[j, k])
            train_label_x_grad_paths.append(T2_x_grad_path_matrix[j, k])
            train_label_y_grad_paths.append(T2_y_grad_path_matrix[j, k])

            l += 1

    valid_T2_mask_img = []
    [valid_T2_mask_img.append(np.load(T2_mask_img_path_matrix[fold, i])[3:, 3:]) for i in range(num_of_slices)]
    valid_T2_mask_img = np.asarray(valid_T2_mask_img)
    #shuffling
    train_T1_mask_img_paths, train_data_hg_img_paths, train_data_hg_x_grad_paths, train_data_hg_y_grad_paths, train_data_sr_img_paths, train_data_sr_x_grad_paths, train_data_sr_y_grad_paths, train_label_img_paths, train_label_x_grad_paths, train_label_y_grad_paths = shuffle(train_T1_mask_img_paths, train_data_hg_img_paths, train_data_hg_x_grad_paths, train_data_hg_y_grad_paths, train_data_sr_img_paths, train_data_sr_x_grad_paths, train_data_sr_y_grad_paths, train_label_img_paths, train_label_x_grad_paths, train_label_y_grad_paths)

    numEpochs = 300
    batch_size = 5
    
    for jj in range(numEpochs):
        print("Running epoch : %d" % jj)
        valid_psnr_img = open('result_files/'+ 'Fold' + str(fold) + '_valid_psnr_img' + '.txt', 'a')
        valid_mse_img = open('result_files/'+ 'Fold' + str(fold) + '_valid_mse_img' + '.txt', 'a')
        valid_mae_img = open('result_files/'+ 'Fold' + str(fold) + '_valid_mae_img' + '.txt', 'a')

        valid_psnr_x_grad = open('result_files/' + 'Fold' + str(fold) + '_valid_psnr_Xgrad' + '.txt', 'a')
        valid_mse_x_grad = open('result_files/' + 'Fold' + str(fold) + '_valid_mse_Xgrad' + '.txt', 'a')
        valid_mae_x_grad = open('result_files/' + 'Fold' + str(fold) + '_valid_mae_Xgrad' + '.txt', 'a')

        valid_psnr_y_grad = open('result_files/' + 'Fold' + str(fold) + '_valid_psnr_Ygrad' + '.txt', 'a')
        valid_mse_y_grad = open('result_files/' + 'Fold' + str(fold) + '_valid_mse_Ygrad' + '.txt', 'a')
        valid_mae_y_grad = open('result_files/' + 'Fold' + str(fold) + '_valid_mae_Ygrad' + '.txt', 'a')

        batch_loss_file = open('result_files/' + 'Fold' + str(fold) + '_batch_loss_file' + '.txt', 'a')
        batch_loss_per_epoch_file = open('result_files/' + 'Fold' + str(fold) + '_batch_loss_per_epoch' + '.txt', 'a')
    
        batch_loss_per_epoch = 0.0
        num_batches = int(len(train_data_hg_img_paths)/batch_size)

        for batch in range(num_batches):
            batch_train_hg = np.zeros((batch_size, resizeTo, resizeTo, 3))
            batch_train_sr = np.zeros((batch_size, resizeTo, resizeTo, 3))
            batch_train_label = np.zeros((batch_size, resizeTo, resizeTo, 4))
            element_in_batch = 0
            for each_npy in range(batch*batch_size, min((batch+1)*batch_size, len(train_T1_mask_img_paths))):
                batch_train_hg[element_in_batch, :, :, 0] = np.load(train_data_hg_img_paths[each_npy])[3:, 3:] * np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                batch_train_hg[element_in_batch, :, :, 1] = np.load(train_data_hg_x_grad_paths[each_npy])[3:, 3:] * np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                batch_train_hg[element_in_batch, :, :, 2] = np.load(train_data_hg_y_grad_paths[each_npy])[3:, 3:] * np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                
                batch_train_sr[element_in_batch, :, :, 0] = np.load(train_data_sr_img_paths[each_npy])[3:, 3:] * np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                batch_train_sr[element_in_batch, :, :, 1] = np.load(train_data_sr_x_grad_paths[each_npy])[3:, 3:] * np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                batch_train_sr[element_in_batch, :, :, 2] = np.load(train_data_sr_y_grad_paths[each_npy])[3:, 3:] * np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                
                batch_train_label[element_in_batch, :, :, 0] = np.load(train_label_img_paths[each_npy])[3:, 3:] * np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                batch_train_label[element_in_batch, :, :, 1] = np.load(train_label_x_grad_paths[each_npy])[3:, 3:] * np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                batch_train_label[element_in_batch, :, :, 2] = np.load(train_label_y_grad_paths[each_npy])[3:, 3:] * np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                batch_train_label[element_in_batch, :, :, 3] = np.load(train_T1_mask_img_paths[each_npy])[3:, 3:]
                
                element_in_batch += 1
 
            loss = model.train_on_batch([batch_train_hg, batch_train_sr],batch_train_label)
            print ('epoch_num: %d batch_num: %d loss: %f\n' % (jj, batch, loss))
            batch_loss_file.write("%d %d %f\n" % (jj, batch, loss))
            batch_loss_per_epoch += loss
        
        batch_loss_per_epoch = batch_loss_per_epoch / num_batches
        batch_loss_per_epoch_file.write("%d %f\n" % (jj, batch_loss_per_epoch))

        if jj == (numEpochs - 1):
            model.save_weights("./Model/" + "FoldNum" + str(fold) + "_LastEpoch.h5")
        if jj % 50 == 0:
            model.save_weights("./Model/" + "FoldNum" + str(fold) + "_EpochNum"+ str(jj) +".h5")
    
        decoded_imgs = model.predict([valid_data_hg, valid_data_sr])
        result_img = decoded_imgs[:,:,:,0] * valid_T1_mask_img
        mse_img =  np.mean((valid_label[:,:,:,0] - result_img) ** 2)
        check_mse = math.sqrt(mse_img)
        psnr_img = 20 * math.log10( 1.0 / (check_mse))
        mae_img = np.mean(np.abs((valid_label[:,:,:,0] - result_img)))

        mse_Xgrad =  np.mean((valid_label[:,:,:,1] - valid_data_hg[:,:,:,1]) ** 2)
        check = math.sqrt(mse_Xgrad)
        psnr_Xgrad = 20 * math.log10( 1.0 / check)
        mae_Xgrad = np.mean(np.abs((valid_label[:,:,:,1] - valid_data_hg[:,:,:,1])))

        mse_Ygrad =  np.mean((valid_label[:,:,:,2] - valid_data_hg[:,:,:,2]) ** 2)
        check = math.sqrt(mse_Ygrad)
        psnr_Ygrad = 20 * math.log10( 1.0 / check)
        mae_Ygrad = np.mean(np.abs((valid_label[:,:,:,2] - valid_data_hg[:,:,:,2])))

        valid_psnr_img.write("%f \n" % (psnr_img))
        valid_mse_img.write("%f \n" % (check_mse))
        valid_mae_img.write("%f \n" %(mae_img))
        
        valid_psnr_x_grad.write("%f \n" % (psnr_Xgrad))
        valid_mse_x_grad.write("%f \n" % (mse_Xgrad))
        valid_mae_x_grad.write("%f \n" %(mae_Xgrad))

        valid_psnr_y_grad.write("%f \n" % (psnr_Ygrad))
        valid_mse_y_grad.write("%f \n" % (mse_Ygrad))
        valid_mae_y_grad.write("%f \n" %(mae_Ygrad))

        if (jj % 50 == 0) or (jj == (numEpochs - 1)):
            print("asdf")
            for slice_num in [20,40,60,80,100,120,140,160,180]:
                temp = np.zeros([resizeTo, resizeTo*7])
                temp[:resizeTo,:resizeTo] = valid_data_hg[slice_num,:,:,0]
                temp[:resizeTo,resizeTo:resizeTo*2] = valid_label[slice_num,:,:,0]
                temp[:resizeTo,resizeTo*2:resizeTo*3] = decoded_imgs[slice_num,:,:, 0]
                temp[:resizeTo,resizeTo*3:resizeTo*4] = valid_T1_mask_img[slice_num,:,:]
                temp[:resizeTo,resizeTo*4:resizeTo*5] = result_img[slice_num,:,:] 
                temp[:resizeTo,resizeTo*5:resizeTo*6] = valid_T2_mask_img[slice_num,:,:]
                temp[:resizeTo,resizeTo*6:] = abs(result_img[slice_num,:,:] - valid_label[slice_num,:,:,0])
                temp = temp * 255
                #scipy.misc.imsave('results_sliced_4layers_custom_loss/' + str(jj) + '.jpg', temp)
                path = os.path.join('./Images', str(slice_num))
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                cv2.imwrite('./Images/' + str(slice_num)+ '/' + "FoldNum" + str(fold) + "_EpochNum"+ str(jj) + "_Slice_num" + str(slice_num) + '.jpg', temp)
        jj +=1

        valid_psnr_img.close()
        valid_mse_img.close()
        valid_mae_img.close()

        valid_psnr_x_grad.close()
        valid_mse_x_grad.close()
        valid_mae_x_grad.close()
        
        valid_psnr_y_grad.close()
        valid_mse_y_grad.close()
        valid_mae_y_grad.close()

        batch_loss_file.close()
        batch_loss_per_epoch_file.close()

        predicted_nii = nib.Nifti1Image(result_img, affine=np.eye(4))
        subject_name = ['08002CHJE', '08027SYBR', '08029IVDI', '08031SEVE', '08037ROGU']
        nib.save(predicted_nii,'./predicted_' + subject_name[fold] + '_Fold' + str(fold) + '.nii.gz')

model = make_model()
train_model(fold = 2)























