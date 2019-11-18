from tf_SDUnet import unet, util, image_util,metrics
import  numpy as np
data_provider2 = image_util.ImageDataProvider("TNBC/Chase1100/test/*",data_suffix="R.jpg", mask_suffix='R_1stHO.png', n_class=2)

net = unet.Unet(layers=4, features_root=8, channels=3, n_class=2)
test_x, test_y = data_provider2(1)
prediction = net.predict("mode/CHASE100_0.93_1100/model.ckpt", test_x)
print(prediction.shape)
prediction=util.crop_to_shape(prediction, (14,960,999,2))

AUC_ROC = metrics.roc_Auc(prediction,util.crop_to_shape(test_y, prediction.shape))
print("auc",AUC_ROC)
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



#
#     print(prediction_tensor)
#     print(label_tensor)
#     print("AUC:" + str(value))


img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "01R.jpg")
