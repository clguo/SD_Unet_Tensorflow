from tf_SDUnet import unet, util, image_util,metrics
import  numpy as np
import  EVA
data_provider2 = image_util.ImageDataProvider("STARE_NEW/test/*", data_suffix="_train.jpg", mask_suffix='_label.png',n_class=2)
net = unet.Unet(layers=4, features_root=8, channels=3, n_class=2)
test_x, test_y = data_provider2(1)

prediction = net.predict("mode/STARE100_40_0.93/model.ckpt", test_x)

print(prediction.shape)
prediction=util.crop_to_shape(prediction, (10,605,700,2))
print(prediction.shape)
print(prediction)


AUC_ROC = metrics.roc_Auc(prediction, util.crop_to_shape(test_y, prediction.shape))
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
util.save_image(img, "0236.jpg")
