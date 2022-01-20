#此訓練模型不是在Raspberry Pi上面，而是在Windows作業系統上
import os
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

image_path = '放入已經分類號類別的圖鑒總檔案的路徑'# example：image_path = 'C:/Users/User/Desktop/Model/sps'
data = DataLoader.from_folder(image_path)
train_data, rest_data = data.split(0.8)# 作爲訓練集
validation_data, test_data = rest_data.split(0.5)# 再把剩餘的分割為驗證集與測試集
model = image_classifier.create(train_data, validation_data=validation_data, epochs = 15)# epochs 訓練組的次數（自訂）
loss, accuracy = model.evaluate(test_data)# 偏差，準度
model.export(export_dir='.',tflite_filename='輸出檔案的名稱.tflite') # example：tflite_filename='sps.tflite'
