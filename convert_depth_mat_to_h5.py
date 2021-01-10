
import os
import h5py
import glob

def convert_mat_to_h5(mat_path, h5_output_path):

    data_w=h5py.File(h5_output_path,'w')
    for sample in glob.glob(path):
        print('sample:',sample)
        imname = sample.split('/')[-2] + '.jpg'
        data=h5py.File(sample,'r')
        data_w.create_dataset(imname, data=data['data_obj'][:])
    data_w.close()

if __name__=='__main__':
    path = r'/data/nfs/yangsuhui/SynthText/prep_scripts/new_own_image_material/results_three/custom_outdoor_sample/*/*.mat'
    output_path = r'/data/nfs/yangsuhui/SynthText/prep_scripts/new_own_image_material/depth.h5'
    convert_mat_to_h5(path,output_path)