import nmrglue as ng
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import nilearn
from nilearn import plotting
from nilearn import image
from nilearn import datasets
import nibabel as nib
import matplotlib.image as mpimg
import h5py
import imageio
import scipy.misc as spmi
import nibabel as nib
from nilearn.image import get_data

func_filenames=[]
for i in range(30):
    func_filenames.append(r'D:\Nilearn_data\development_fmri'+'\sub-pixar'+'{:03d}'.format(i+1) +'_task-pixar_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')

confounds=[]
for i in range(30):
    confounds.append(r'D:\Nilearn_data\development_fmri'+'\sub-pixar'+'{:03d}'.format(i+1) +'_task-pixar_desc-reducedConfounds_regressors.tsv')

func_filenames1=[]
for i in range(30):
    func_filenames1.append(r'D:\Nilearn_data\development_fmri'+'\sub-pixar'+'{:03d}'.format(i+123) +'_task-pixar_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')

confounds1=[]
for i in range(30):
    confounds1.append(r'D:\Nilearn_data\development_fmri'+'\sub-pixar'+'{:03d}'.format(i+123) +'_task-pixar_desc-reducedConfounds_regressors.tsv')

#########################################################################


for i in range(10):
    img = nib.load(func_filenames1[i])
    img_data = img.get_fdata()
    
    for j in range(168):
        
        for k in range(8,36):
            brain_slice= img_data[:,:,k,j]
            imageio.imwrite('adult'+'{:02d}'.format(i)+'{:03d}'.format(j)+'{:02d}'.format(k)+'.jpg', brain_slice)


###############################

# Get nibabel image object
img = nib.load(func_filenames1[0])

# Get data from nibabel image object (returns numpy memmap object)
img_data = img.get_fdata()

brain_slice= img_data[5,:,:,1]

for i in range(6,15):
    brain_slice= img_data[:,:,i,1]
    plt.figure(str(i))
    plt.imshow(brain_slice,cmap='gray')


plt.imshow(brain_slice,cmap='gray')

img = mpimg.imread('child0000034.png')
im = imageio.imread('child0000034.png')



# Convert to numpy ndarray (dtype: uint16)
img_data_arr = np.asarray(img_data)
##################################

b=np.array(a)



first_rsn = image.index_img('nitin.nii', 1)

b=get_data(first_rsn)



for i in range(30):
    image_child=nilearn.image.get_data(nilearn.image.load_img(func_filenames[i]))
    for j in range(168):
        np.save('child'+'{:02d}'.format(i)+'{:03d}'.format(j), image_child[:,:,:,j].reshape(50,59,50,1))
 
for i in range(22,30):
    image_adult=nilearn.image.get_data(nilearn.image.load_img(func_filenames1[i]))
    for j in range(168):
        np.save('adult'+'{:02d}'.format(i)+'{:03d}'.format(j), image_adult[:,:,:,j].reshape(50,59,50,1))
 
        
from PIL import Image as pimg
newImg1 = pimg.new('RGB', image_adult[:,:,:,1].reshape(50,59,50,1))
newImg1.save("img1.png")
    
    
    
    
img=nilearn.image.get_data(nilearn.image.load_img(func_filenames[0]))

img1=nib.Nifti1Image(img[:,:,:,0].reshape(50,59,50,1),np.eye(4))

img1.to_filename('test.nii')


anat_img_data = anat_img.get_fdata()


anat_img = nib.load(func_filenames[i])
anat_img.affine






# Don't forget to close hdf file
hdf= h5py.File('dev_CNN.h5','w')

G1= hdf.create_group('training_set')
G1.create_dataset('child')



    
#    data_adult1.append(nilearn.image.get_data(nilearn.image.load_img(func_filenames1[i])))


data_child1.shape



data_child2=[]
data_adult2=[]
for i in np.linspace(16,29,14,dtype=np.int32):
    data_child2.append(nilearn.image.get_data(nilearn.image.load_img(func_filenames[i])))
    data_adult2.append(nilearn.image.get_data(nilearn.image.load_img(func_filenames1[i])))


for i in np.linspace(16,29,14,dtype=np.int32):
    print(i)

for i in range(15):
    data_child.append(nilearn.image.get_data(nilearn.image.load_img(func_filenames[i])))
    data_adult.append(nilearn.image.get_data(nilearn.image.load_img(func_filenames1[i])))




data_child=nilearn.image.get_data(images_child)
data_adult=nilearn.image.get_data(images_adult)

X=[]

for i in np.linspace(0,5020,252):
    X.append(np.concatenate((data_child[:,:,:,i:(i+20)],data_adult[:,:,:,i:(i+20)]),axis=3))




data_child = nib.load(images_child) 
data = img.get_data()
data.shape

vol_shape = data.shape[:-1]
n_voxels = np.prod(vol_shape)

voxel_by_time = data.reshape(n_voxels, data.shape[-1])
voxel_by_time.shape

voxel_by_time= data[:,:,:,0]

#voxel_by_time=voxel_by_time.reshape(50,59,1)

brain_slice= voxel_by_time[:,:,30]
plt.imshow(brain_slice,cmap='gray')
