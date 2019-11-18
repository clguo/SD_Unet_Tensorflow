# -*- coding: utf-8 -*-
"""
@author: tz_zs
图片的编码、解码、随机截取图像
"""

import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("data_set/label/0.tif", 'rb').read()

with tf.Session() as sess:
    # 解码
    image_data = tf.image.decode_jpeg(image_raw_data)

    print(image_data.eval())


