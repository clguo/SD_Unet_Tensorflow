from tf_SDUnet import unet, util, image_util

#preparing data loading
data_provider = image_util.ImageDataProvider("dataset2/train/*.jpg", data_suffix=".jpg", mask_suffix='_mask.png', n_class=2)
output_path="out_put2"
#setup & training
net = unet.Unet(layers=6, features_root=16, channels=3, n_class=2)

trainer = unet.Trainer(net)
path = trainer.train(data_provider, output_path, training_iters=12, epochs=1)
test_x, test_y = data_provider(1)
print(test_x.shape)
prediction = net.predict(path, test_x)
print(prediction)
unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "voc_prediction.jpg")