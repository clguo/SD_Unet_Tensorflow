from tf_SDUnet import unet, util, image_util
data_provider2 = image_util.ImageDataProvider("TNBC/test/*.png",data_suffix=".png", mask_suffix='_mask.png', n_class=2)
#setup & training
net = unet.Unet(layers=4, features_root=6, channels=3, n_class=2)

test_x, test_y = data_provider2(25)

prediction = net.predict("TNBC2_CKPT/model.ckpt", test_x)

print(test_y)
error=unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print("test error:",error)
img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "TNBC_test13.jpg")