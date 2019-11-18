
from tf_SDUnet import image_gen,image_util,unet2,util
import  numpy as np

#preparing data loading
# data_provider = image_util.ImageDataProvider("nuclei/*.png",data_suffix=".png", mask_suffix=' (2).png', n_class=2)
#
# data_provider = image_util.ImageDataProvider("data_set/train/*.tif",data_suffix=".tif", mask_suffix='_mask.tif', n_class=2)
data_provider2 = image_util.ImageDataProvider("test/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
#data_provider = image_util.ImageDataProvider("Tissue_images/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
data_provider = image_util.ImageDataProvider("tissue_aug/train/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
output_path="out_put_skip_aug3_160"
#setup & training
net = unet2.Unet(layers=4, features_root=8, channels=3, n_class=2)
trainer = unet2.Trainer(net,batch_size=4,verification_batch_size=4,optimizer="adam")
path = trainer.train(data_provider, output_path,keep_prob=0.8,training_iters=124, epochs=1,display_step=2,restore=True)
test_x, test_y = data_provider2(14)
prediction = net.predict(path, test_x)

print(test_y)
error=unet2.error_rate(prediction, util.crop_to_shape(test_y, prediction.shape))
print("test error:",error)
img = util.combine_img_prediction(test_x, test_y, prediction)
util.save_image(img, "test_aug_5.jpg")