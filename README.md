# INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE
Intracranial Hemorrhage Segmentation Challenge on Non-Contrast head CT (NCCT)
> This is an example of the CT imaging is used to Segment Intracranial Hemorrhage Segmentation.
![](INSTANCE2022.png)

## Prerequisities
The following dependencies are needed:
- numpy >= 1.11.1
- SimpleITK >=1.0.1
- pytorch-gpu ==1.10.0
- pandas >=0.20.1
- scikit-learn >= 0.17.1

## How to Use
* 1、when download the all project,check out the data folder all csv,put your train data into same folder.or you can run Instancedata3dpreparewithSize.py to generate train data and validation data.
* 2、run Instance_train.py for Unet3d/Vnet3d segmeatation training:make sure train data have effective path
* 3、run Instance_inference.py for Unet3d/Vnet3d segmeatation inference:make sure test data have effective path

## Result

# Unet3d
* dice:train loss,train accuracy,validation loss,validation accuracy
![](Unet_train_loss_valdation_lossplot.png)
![](Unet_train_accuracy_valdation_accuracyplot.png)

# Vnet3d
* dice:train loss,train accuracy,validation loss,validation accuracy
![](Vnet_train_loss_valdation_lossplot.png)
![](Vnet_train_accuracy_valdation_accuracyplot.png)

# test dataset result
* test dataset segmentation result
![](1.png)
![](2.png)
![](3.png)
![](4.png)
![](5.png)
* test dataset leadboard
![](leadboard.png)
![](test_metric.png)

* you can find the trained model in log folder and test predict mask with zip file.
* more detail and trained model can follow my WeChat Public article.

## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com
* Contact: junqiangChen
* WeChat Number: 1207173174
* WeChat Public number: 最新医学影像技术
