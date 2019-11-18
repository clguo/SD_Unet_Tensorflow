
from tf_SDUnet import FCN,image_util,unet,util

data_provider2 = image_util.ImageDataProvider("test/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
data_provider = image_util.ImageDataProvider("Tissue_images/*.tif",data_suffix=".tif", mask_suffix='_binary.tif', n_class=2)
output_path="FCN_put"
#setup & training
net = FCN.fcn(channels=3, n_class=2)
trainer = FCN.Trainer(net,batch_size=4,verification_batch_size=4,optimizer="adam")
path = trainer.train(data_provider, output_path,dropout=0.9,training_iters=32, epochs=20,display_step=2)
test_x, test_y = data_provider2(1)
prediction = net.predict(path, test_x)
print(prediction)
print(test_y)
error=FCN.error_rate(prediction, test_y)
print("test error:",error)
img = util.combine_img_predictionFCN(test_x, test_y, prediction)
util.save_image(img, "FCN.jpg")