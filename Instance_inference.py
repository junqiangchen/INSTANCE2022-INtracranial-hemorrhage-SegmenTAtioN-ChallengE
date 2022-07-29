import torch
import os
from model import *
from dataprocess.utils import file_name_path, MorphologicalOperation, GetLargestConnectedCompont, \
    GetLargestConnectedCompontBoundingbox
import SimpleITK as sitk
import numpy as np

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def inferencemutilunet3dtest():
    newSize = (256, 320, 32)
    Unet3d = BinaryUNet3dModel(image_depth=32, image_height=320, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss', inference=True,
                               model_path=r'log\instance\dice\Unet\BinaryUNet3d.pth')
    datapath = r"D:\challenge\data\Instance2022\evaluation"
    makspath = r"D:\challenge\data\Instance2022\predict\Unet"
    image_path_list = file_name_path(datapath, False, True)
    for i in range(len(image_path_list)):
        imagepathname = datapath + "/" + image_path_list[i]
        src = sitk.ReadImage(imagepathname)
        binary_src = sitk.BinaryThreshold(src, 100, 5000)
        binary_src = MorphologicalOperation(binary_src, 1)
        binary_src = GetLargestConnectedCompont(binary_src)
        boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
        # print(boundingbox)  # (x,y,z,xlength,ylength,zlength)
        x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], \
                                 boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
        src_array = sitk.GetArrayFromImage(src)
        roi_src_array = src_array[z1:z2, y1:y2, x1:x2]
        roi_src = sitk.GetImageFromArray(roi_src_array)
        roi_src.SetSpacing(src.GetSpacing())
        roi_src.SetDirection(src.GetDirection())
        roi_src.SetOrigin(src.GetOrigin())

        sitk_mask = Unet3d.inference(roi_src, newSize)

        roi_binary_array = sitk.GetArrayFromImage(sitk_mask)
        binary_array = np.zeros_like(src_array)
        binary_array[z1:z2, y1:y2, x1:x2] = roi_binary_array[:, :, :]
        binary_vessels = sitk.GetImageFromArray(binary_array)
        binary_vessels.SetSpacing(src.GetSpacing())
        binary_vessels.SetDirection(src.GetDirection())
        binary_vessels.SetOrigin(src.GetOrigin())

        maskpathname = makspath + "/" + image_path_list[i]
        sitk.WriteImage(binary_vessels, maskpathname)


def inferencemutilvnet3dtest():
    newSize = (256, 320, 32)
    vnet3d = BinaryVNet3dModel(image_depth=32, image_height=320, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss', inference=True,
                               model_path=r'log\instance\dice\Vnet\BinaryVNet3d.pth')
    datapath = r"D:\challenge\data\Instance2022\evaluation"
    makspath = r"D:\challenge\data\Instance2022\predict\Vnet"
    image_path_list = file_name_path(datapath, False, True)
    for i in range(len(image_path_list)):
        imagepathname = datapath + "/" + image_path_list[i]
        src = sitk.ReadImage(imagepathname)
        binary_src = sitk.BinaryThreshold(src, 100, 5000)
        binary_src = MorphologicalOperation(binary_src, 1)
        binary_src = GetLargestConnectedCompont(binary_src)
        boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
        # print(boundingbox)  # (x,y,z,xlength,ylength,zlength)
        x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], \
                                 boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
        src_array = sitk.GetArrayFromImage(src)
        roi_src_array = src_array[z1:z2, y1:y2, x1:x2]
        roi_src = sitk.GetImageFromArray(roi_src_array)
        roi_src.SetSpacing(src.GetSpacing())
        roi_src.SetDirection(src.GetDirection())
        roi_src.SetOrigin(src.GetOrigin())

        sitk_mask = vnet3d.inference(roi_src, newSize)

        roi_binary_array = sitk.GetArrayFromImage(sitk_mask)
        binary_array = np.zeros_like(src_array)
        binary_array[z1:z2, y1:y2, x1:x2] = roi_binary_array[:, :, :]
        binary_vessels = sitk.GetImageFromArray(binary_array)
        binary_vessels.SetSpacing(src.GetSpacing())
        binary_vessels.SetDirection(src.GetDirection())
        binary_vessels.SetOrigin(src.GetOrigin())

        maskpathname = makspath + "/" + image_path_list[i]
        sitk.WriteImage(binary_vessels, maskpathname)


if __name__ == '__main__':
    # inferencemutilunet3dtest()
    inferencemutilvnet3dtest()
