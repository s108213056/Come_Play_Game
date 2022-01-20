import os
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

image_path = 'C:/Users/User/Desktop/Model/sps'
data = DataLoader.from_folder(image_path)
train_data, rest_data = data.split(0.8)#訓練集
validation_data, test_data = rest_data.split(0.5)#把分割0.2再分攤一半
model = image_classifier.create(train_data, validation_data=validation_data, epochs = 15)
loss, accuracy = model.evaluate(test_data)#偏差，準度
model.export(export_dir='.',tflite_filename='sps2.tflite')