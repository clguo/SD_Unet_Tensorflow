
from tf_SDUnet import image_gen,image_util,unet,util
import  numpy as np

#preparing data loading
# data_provider = image_util.ImageDataProvider("nuclei/*.png",data_suffix=".png", mask_suffix=' (2).png', n_class=2)
#
# data_provider = image_util.ImageDataProvider("data_set/train/*.tif",data_suffix=".tif", mask_suffix='_mask.tif', n_class=2)
DT = image_util.ImageDataProvider("test2/DT/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
ST = image_util.ImageDataProvider("test2/ST/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
# data_provider = image_util.ImageDataProvider("Tissue_images/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
#data_provider = image_util.ImageDataProvider("Kumar_aug/aug/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
data_provider = image_util.ImageDataProvider("tissue_aug/train/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
output_path="model_f8_0.88_dice_100"
#setup & training
net = unet.Unet(layers=4, features_root=8, channels=3, n_class=2,cost="dice_coefficient")
trainer = unet.Trainer(net,batch_size=4,verification_batch_size=4,optimizer="adam")
path = trainer.train(data_provider, output_path,keep_prob=0.88,block_size=7,training_iters=64, epochs=100,display_step=2,restore=False)
test_x, test_y = DT(6)
prediction = net.predict(path, test_x)
error=unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print("DT error:",error)
f1=unet.f1score2(prediction, util.crop_to_shape(test_y, prediction.shape))
print("DTf1:",f1)
img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "DTtest.jpg")
test_x, test_y = ST(8)
prediction = net.predict(path, test_x)
error=unet.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print("ST error:",error)
f1=unet.f1score2(prediction, util.crop_to_shape(test_y, prediction.shape))
print("STf1:",f1)
img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "STtest.jpg")