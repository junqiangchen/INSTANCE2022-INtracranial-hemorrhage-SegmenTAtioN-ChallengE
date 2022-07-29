from __future__ import print_function, division

import os
import SimpleITK as sitk
import numpy as np
from dataprocess.utils import ConvertitkTrunctedValue, resize_image_itkwithsize
from dataprocess.utils import file_name_path, GetLargestConnectedCompont, GetLargestConnectedCompontBoundingbox, \
    MorphologicalOperation

image_dir = "Image"
mask_dir = "Mask"
image_pre = ".nii.gz"
mask_pre = ".nii.gz"


def preparesampling3dtraindata(datapath, trainImage, trainMask, shape=(96, 96, 96)):
    newSize = shape
    dataImagepath = datapath + "/" + image_dir
    dataMaskpath = datapath + "/" + mask_dir
    all_files = file_name_path(dataImagepath, False, True)
    for subsetindex in range(len(all_files)):
        mask_name = all_files[subsetindex]
        mask_gt_file = dataMaskpath + "/" + mask_name
        masksegsitk = sitk.ReadImage(mask_gt_file, sitk.sitkUInt8)
        image_name = all_files[subsetindex]
        image_gt_file = dataImagepath + "/" + image_name
        imagesitk = sitk.ReadImage(image_gt_file, sitk.sitkInt16)

        _, resizeimage = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        resizeimage = ConvertitkTrunctedValue(resizeimage, 100, 0, 'meanstd')
        resizemaskarray, resizemask = resize_image_itkwithsize(masksegsitk, newSize, masksegsitk.GetSize(),
                                                               sitk.sitkNearestNeighbor)
        resizeimagearray = sitk.GetArrayFromImage(resizeimage)
        # step 3 get subimages and submasks
        if not os.path.exists(trainImage):
            os.makedirs(trainImage)
        if not os.path.exists(trainMask):
            os.makedirs(trainMask)
        filepath1 = trainImage + "\\" + str(subsetindex) + ".npy"
        filepath = trainMask + "\\" + str(subsetindex) + ".npy"
        np.save(filepath1, resizeimagearray)
        np.save(filepath, resizemaskarray)


def ProcessBrainROI():
    train_path = r"F:\MedicalData\2022Instance\train"
    train_image_path = train_path + '/' + 'data'
    train_label_path = train_path + '/' + 'label'
    roi_train_path = r"F:\MedicalData\2022Instance\ROIProcess\train"
    trainimage_list = file_name_path(train_image_path, False, True)

    for subsetindex in range(len(trainimage_list)):
        image_name = trainimage_list[subsetindex]
        image_file = train_image_path + "/" + image_name
        src = sitk.ReadImage(image_file, sitk.sitkInt16)
        mask_file = train_label_path + "/" + image_name
        mask = sitk.ReadImage(mask_file, sitk.sitkUInt8)
        binary_src = sitk.BinaryThreshold(src, 100, 5000)
        binary_src = MorphologicalOperation(binary_src, 1)
        binary_src = GetLargestConnectedCompont(binary_src)
        boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
        # print(boundingbox)  # (x,y,z,xlength,ylength,zlength)
        x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], \
                                 boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
        src_array = sitk.GetArrayFromImage(src)
        mask_array = sitk.GetArrayFromImage(mask)
        print(np.unique(mask_array))
        roi_src_array = src_array[z1:z2, y1:y2, x1:x2]
        roi_mask_array = mask_array[z1:z2, y1:y2, x1:x2]
        roi_src = sitk.GetImageFromArray(roi_src_array)
        roi_src.SetSpacing(src.GetSpacing())
        roi_src.SetDirection(src.GetDirection())
        roi_src.SetOrigin(src.GetOrigin())
        roi_mask = sitk.GetImageFromArray(roi_mask_array)
        roi_mask.SetSpacing(mask.GetSpacing())
        roi_mask.SetDirection(mask.GetDirection())
        roi_mask.SetOrigin(mask.GetOrigin())
        image_file = roi_train_path + "/Image/" + image_name
        sitk.WriteImage(roi_src, image_file)
        mask_file = roi_train_path + "/Mask/" + image_name
        sitk.WriteImage(roi_mask, mask_file)


def preparetraindata():
    """
    :return:
    """
    src_train_path = r"D:\challenge\data\Instance2022\ROIprocess\train"
    source_process_path = r"D:\challenge\data\Instance2022\trainstage\train"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, (256, 320, 32))


def preparevalidationdata():
    """
    :return:
    """
    src_train_path = r"D:\challenge\data\Instance2022\ROIprocess\validation"
    source_process_path = r"D:\challenge\data\Instance2022\trainstage\validation"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, (256, 320, 32))


if __name__ == "__main__":
    # ProcessBrainROI()
    preparetraindata()
    preparevalidationdata()
