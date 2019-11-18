from tf_SDUnet import unet, util, image_util,metrics
import  EVA
import  numpy as np

import os
# data_provider2 = image_util.ImageDataProvider("CHASE_NEW/test/*",data_suffix="R.jpg", mask_suffix='R_1stHO.png', n_class=2)
# data_provider2 = image_util.ImageDataProvider("Drive_NEW/test/*",data_suffix="_test.tif", mask_suffix='_manual1.png', n_class=2)
data_provider2 = image_util.ImageDataProvider("STARE_NEW/test/*",data_suffix="_train.jpg", mask_suffix='_label.png', n_class=2)
#data_provider2 = image_util.ImageDataProvider("DriveTest1/*",data_suffix="_test.tif", mask_suffix='_manual1.gif', n_class=2)
data_provider = image_util.ImageDataProvider("STARE_NEW/train/*",data_suffix="_train.jpg", mask_suffix='_label.png', n_class=2)
output_path="mode/STARE100_0.93"
net = unet.Unet(layers=4, features_root=8, channels=3, n_class=2)
trainer = unet.Trainer(net,batch_size=4,verification_batch_size=4,optimizer="adam")
path = trainer.train(data_provider, output_path,keep_prob=0.93,training_iters=40, epochs=100,display_step=2,restore=False)
test_x, test_y = data_provider2(10)

prediction = net.predict(path, test_x)
prediction=util.crop_to_shape(prediction, (10,605,700,2))
AUC_ROC = metrics.roc_Auc(prediction, util.crop_to_shape(test_y, prediction.shape))
print("auc:",AUC_ROC)
acc=metrics.acc(prediction,util.crop_to_shape(test_y, prediction.shape))
print("acc:",acc)
precision=metrics.precision(prediction,util.crop_to_shape(test_y, prediction.shape))
print("ppv:",precision)
sen=metrics.sen(prediction,util.crop_to_shape(test_y, prediction.shape))
print("TPR:",sen)
TNR=metrics.TNR(prediction,util.crop_to_shape(test_y, prediction.shape))
print("tnr:",TNR)
f1=metrics.f1score2(prediction,util.crop_to_shape(test_y, prediction.shape))
print("f1:",f1)
img = util.combine_img_prediction(test_x, test_y, prediction)

util.save_image(img, "STARE.jpg")