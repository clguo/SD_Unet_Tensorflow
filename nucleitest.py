from tf_SDUnet import unet, util, image_util,stats_utils,layers
import tensorflow as tf
import numpy as np
DT = image_util.ImageDataProvider("test2/DT/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2,shuffle_data=False)
ST = image_util.ImageDataProvider("test2/ST/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2,shuffle_data=False)
ALL= image_util.ImageDataProvider("test2/all/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2,shuffle_data=False)
net = unet.Unet(layers=4, features_root=8, channels=3, n_class=2)
test_x, test_y = DT(6)
model="model/model_f8_0.88_flip_160/model.ckpt"
prediction = net.predict(model, test_x)
#
# pred=tf.cast(prediction, tf.float64)
# loss=stats_utils.get_dice(util.crop_to_shape(test_y, prediction.shape),prediction)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     # Initialize variables
#     sess.run(init)
#     print("loss",sess.run(loss))
print("dice2:",stats_utils.get_dice_2(util.crop_to_shape(test_y, prediction.shape),prediction))
print("dice1",stats_utils.get_dice_1(util.crop_to_shape(test_y, prediction.shape),prediction))
print("f1:",unet.f1score2(prediction, util.crop_to_shape(test_y, prediction.shape)))
print("aji:",stats_utils.get_aji(util.crop_to_shape(test_y, prediction.shape),prediction))
print("aji+:",stats_utils.get_fast_aji(util.crop_to_shape(test_y, prediction.shape),prediction))
# DQ,SQ=stats_utils.get_fast_panoptic_quality(util.crop_to_shape(test_y, prediction.shape),prediction)
# print("DQ:",DQ)
# print("SQ:",SQ)
# print("PQ:",DQ*SQ)
error=unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print("DT error:",error)
# f1=unet.f1score2(prediction, util.crop_to_shape(test_y, prediction.shape))
# print("DTf1:",f1)
img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "DTtest.jpg")

test_x, test_y = ST(8)
prediction = net.predict(model, test_x)

# #
print("dice:",stats_utils.get_dice_2(util.crop_to_shape(test_y, prediction.shape),prediction))
print("aji:",stats_utils.get_aji(util.crop_to_shape(test_y, prediction.shape),prediction))
print("aji+:",stats_utils.get_fast_aji(util.crop_to_shape(test_y, prediction.shape),prediction))
DQ,SQ=stats_utils.get_fast_panoptic_quality(util.crop_to_shape(test_y, prediction.shape),prediction)
print("DQ:",DQ)
print("SQ:",SQ)
print("PQ:",DQ*SQ)
error=unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print("ST error:",error)

test_x, test_y = ALL(14)
prediction = net.predict(model, test_x)
print("dice:",stats_utils.get_dice_2(util.crop_to_shape(test_y, prediction.shape),prediction))
print("aji:",stats_utils.get_aji(util.crop_to_shape(test_y, prediction.shape),prediction))
print("aji+:",stats_utils.get_fast_aji(util.crop_to_shape(test_y, prediction.shape),prediction))
DQ,SQ=stats_utils.get_fast_panoptic_quality(util.crop_to_shape(test_y, prediction.shape),prediction)
print("DQ:",DQ)
print("SQ:",SQ)
print("PQ:",DQ*SQ)
error=unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print("AVEerror:",error)
img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "alltest.jpg")