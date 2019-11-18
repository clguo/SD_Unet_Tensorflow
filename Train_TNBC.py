from tf_SDUnet import unet, util, image_util

import  numpy as np

import os

data_provider2 = image_util.ImageDataProvider("TNBC/test/*.png",data_suffix=".png", mask_suffix='_mask.png', n_class=2)
data_provider = image_util.ImageDataProvider("Model_zoo/train_aug/train/*.png",data_suffix=".png", mask_suffix='_mask.png', n_class=2)
output_path="TNBC2_CKPT"
net = unet.Unet(layers=4, features_root=6, channels=3, n_class=2)
trainer = unet.Trainer(net,batch_size=8,verification_batch_size=4,optimizer="adam")
path = trainer.train(data_provider, output_path,keep_prob=0.8,training_iters=32, epochs=100,display_step=2,restore=True)
test_x, test_y = data_provider2(1)
prediction = net.predict(path, test_x)
error=unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print(error)
img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "tnbc2.jpg")