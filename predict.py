import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
#若是要以Windows作業系統開啓
#import tensorflow as tf
#interpreter = tf.lite.Interpreter(model_path='放入自己訓練模型的路徑')# example：(model_path='sps.tflite')
interpreter = tflite.Interpreter(model_path='放入自己訓練模型的路徑')# example：(model_path='sps.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_path='放入自己要測試的圖片路徑'# example：image_path='s.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img,(224,224))# check.py 顯示的大小（長寬）

input_shape = input_details[0]['shape']
input_tensor= np.array(np.expand_dims(img,0))
input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index, input_tensor)
interpreter.invoke()

output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
pred = np.squeeze(output_data)
print(pred)# 這裏印出訓練模型的比例（255分配）[ 33 216   7]

class_ind = {0:"Paper",1:"Scissors",2:"Stone"}# 列出訓練有幾種種類
highest_pred_loc = np.argmax(pred)# 最大值就是預測結果[0~255]
hand_name = class_ind[highest_pred_loc]
print(hand_name)# 印出最高值的 example：Scissors
