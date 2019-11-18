from tf_SDUnet import unet, util, image_util,F1test
import numpy as  np
import tensorflow as tf

data_provider2 = image_util.ImageDataProvider("test/ST/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2,shuffle_data=False)

#setup & training
net = unet.Unet(layers=4, features_root=8, channels=3, n_class=2)

test_x, test_y = data_provider2(14)

prediction = net.predict("model_aug_f8_0.75_100/model.ckpt", test_x)

print(test_y)
error=unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print("test error:",error)
f1=unet.f1score2(prediction,util.crop_to_shape(test_y, prediction.shape))
# print(f1)
# y_pred = np.argmax(prediction, axis=3)[0]
# y_true = np.argmax(util.crop_to_shape(test_y, prediction.shape), axis=3)[0]
# precision=F1test.precision(y_true,y_pred)
#
# f1=F1test.f1score(y_true,y_pred)


print("f1",f1)

img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "test.jpg")
