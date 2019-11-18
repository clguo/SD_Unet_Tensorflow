from tf_SDUnet import unet, util, image_util,metrics
import  numpy as np
data_provider= image_util.ImageDataProvider("DRIVE700/test/*",data_suffix="_test.tif", mask_suffix='_manual1.png', n_class=2)

net = unet.Unet(layers=4, features_root=8, channels=3, n_class=2)
test_x, test_y = data_provider(1)

prediction = net.predict("mode/drive100_0.92_700/model.ckpt", test_x)
prediction=util.crop_to_shape(prediction, (20,584,565,2))

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


img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "19.jpg")
