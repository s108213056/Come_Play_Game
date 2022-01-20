import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='放入自己訓練出來的模型的路徑')# example：(model_path='sps.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])# 物件名字
print("shape:", input_details[0]['shape'])# [  1 224 224   3] 一個輸出結果；圖片長度壓縮像素；圖片寬度壓縮像素；X個種類的分類
print("type:", input_details[0]['dtype'])# <class 'numpy.uint8'> 8-bit unsigned integer (0 to 255)

print("\n== Output details ==")
print("name:", output_details[0]['name'])# 物件名字
print("shape:", output_details[0]['shape'])# [1 3] 一個輸出結果；X個種類的分類
print("type:", output_details[0]['dtype'])# <class 'numpy.uint8'> 8-bit unsigned integer (0 to 255)
