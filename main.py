import numpy as np
import os
import SimpleITK as sitk
from keras.models import model_from_json
import argparse
parser = argparse.ArgumentParser(description='Detecting AD landmarks')
parser.add_argument('--input', type=str, default='Img.nii.gz', required=False, help='Image path')
parser.add_argument('--outfolder',type=str,default='Results/',required=False,help='Folder for saving results')
opt = parser.parse_args()
print(opt)

def imgnorm(N_I,index):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[index]
    I_max = I_sort[-index]
    
    N_I =255.0*(N_I-I_min)/(I_max-I_min)
    N_I[N_I>255]=255
    N_I2 = N_I.astype(np.uint8)
    
    return N_I2

def Patches(image,sizeofchunk, sizeofchunk_expand,numofchannels):

    sizeofimage = np.shape(image)[1:4]
            
    nb_chunks = (np.ceil(np.array(sizeofimage)/float(sizeofchunk))).astype(int)
    
    pad_image = np.zeros(([numofchannels,nb_chunks[0]*sizeofchunk,nb_chunks[1]*sizeofchunk,nb_chunks[2]*sizeofchunk]), dtype='float32')
    pad_image[:,:sizeofimage[0], :sizeofimage[1], :sizeofimage[2]] = image
            
    width = int(np.ceil((sizeofchunk_expand-sizeofchunk)/2.0))
            
    size_pad_im = np.shape(pad_image)[1:4]
    size_expand_im = np.array(size_pad_im) + 2 * width
    expand_image = np.zeros(([numofchannels,size_expand_im[0],size_expand_im[1],size_expand_im[2]]), dtype='float32')
    expand_image[:,width:-width, width:-width, width:-width] = pad_image
  
            
    batchsize = np.prod(nb_chunks)
    idx_chunk = 0
    chunk_batch = np.zeros((batchsize,numofchannels,sizeofchunk_expand,sizeofchunk_expand,sizeofchunk_expand),dtype='float32')
    idx_xyz = np.zeros((batchsize,3),dtype='uint16')
    for x_idx in range(nb_chunks[0]):
        for y_idx in range(nb_chunks[1]):
            for z_idx in range(nb_chunks[2]):
                
                idx_xyz[idx_chunk,:] = [x_idx,y_idx,z_idx]
                        
                         

                chunk_batch[idx_chunk,:,...] = expand_image[:,x_idx*sizeofchunk:x_idx*sizeofchunk+sizeofchunk_expand,\
                           y_idx*sizeofchunk:y_idx*sizeofchunk+sizeofchunk_expand,\
                           z_idx*sizeofchunk:z_idx*sizeofchunk+sizeofchunk_expand]             
                        
                idx_chunk += 1
    
    return chunk_batch, nb_chunks, idx_xyz, sizeofimage                        

def Images(segment_chunks, nb_chunks, sizeofchunk, sizeofchunk_expand, idx_xyz, sizeofimage):
    
    batchsize = np.size(segment_chunks,0)
    
    segment_image = np.zeros((nb_chunks[0]*sizeofchunk,nb_chunks[1]*sizeofchunk,nb_chunks[2]*sizeofchunk))
    
    for idx_chunk in range(batchsize):
        
        idx_low = idx_xyz[idx_chunk,:] * sizeofchunk
        idx_upp = (idx_xyz[idx_chunk,:]+1) * sizeofchunk
        
        segment_image[idx_low[0]:idx_upp[0],idx_low[1]:idx_upp[1],idx_low[2]:idx_upp[2]] = \
        segment_chunks[idx_chunk,0,...]
        

    segment_image = segment_image[:sizeofimage[0], :sizeofimage[1], :sizeofimage[2]]
    return segment_image

modelpath = 'Model/'
resultpath = opt.outfolder
datapath = opt.input
if not os.path.exists(resultpath):
    os.makedirs(resultpath)
# get model
numofchannels=1
numoflandmarks=50
sizeofchunk =48
sizeofchunk_expand = 94

sizeofpatch=94

json_file = open(modelpath+'architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
Model = model_from_json(loaded_model_json)
# load weights into new model

print 'loading weights ...'
Model.load_weights(modelpath+'Model_Landmark.hd5')

image = np.array(sitk.GetArrayFromImage(sitk.ReadImage(datapath)))


image = imgnorm(image,1000)
image = (image-np.mean(image))/np.std(image)  
    
imagesize = np.shape(image)
image_one = np.zeros((numofchannels,imagesize[0],imagesize[1],imagesize[2]),dtype='float32')
image_one[0,...] = 1.0*image




chunk_batch, nb_chunks, idx_xyz, sizeofimage = Patches(image_one,sizeofchunk,sizeofchunk_expand,numofchannels)

seg_batch = Model.predict(chunk_batch*1.0, batch_size=1)
landmarks = []
for i_landmark in range(numoflandmarks):
    prob_image = Images(seg_batch[:,i_landmark:i_landmark+1,...], nb_chunks, sizeofchunk, sizeofchunk_expand, idx_xyz, sizeofimage)
    
    center_index = np.argmax(prob_image)
    center = np.array(np.unravel_index(center_index,np.shape(prob_image)))
    landmarks.append(center[::-1])
    print center
np.save(resultpath+'landmarks.npy', landmarks)
