from tf_SDUnet import unet, util, image_util,stats_utils,layers
import tensorflow as tf
import numpy as np
Single = image_util.ImageDataProvider("test2/single/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2,shuffle_data=False)
net = unet.Unet(layers=4, features_root=8, channels=3, n_class=2)
test_x, test_y =Single(1)
model="model/model_f8_0.88_flip_160/model.ckpt"
prediction = net.predict(model, test_x)
print("dice2:",stats_utils.get_dice_2(util.crop_to_shape(test_y, prediction.shape),prediction))
print("dice1",stats_utils.get_dice_1(util.crop_to_shape(test_y, prediction.shape),prediction))
print("f1:",unet.f1score2(prediction, util.crop_to_shape(test_y, prediction.shape)))
print("aji:",stats_utils.get_aji(util.crop_to_shape(test_y, prediction.shape),prediction))
print("aji+:",stats_utils.get_fast_aji(util.crop_to_shape(test_y, prediction.shape),prediction))

error=unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print("error:",error)
img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "result/5698.jpg")