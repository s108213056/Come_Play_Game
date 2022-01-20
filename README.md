# 來玩遊戲吧


### 概念發想
- 由於宿舍的四個室友其中一位是被安排進來的，而他的朋友們經常登門造訪，導致宿舍經常有外人進出。於是，我們打算用一個遊戲機制來控制進門的人數!
- ****流程****
1. 進入者須在門口輸入數字"1", 讓pi camera為進入者拍照(拍人臉) / 或者輸入特定指令"7414"，使樹梅派直接讓磁力所失去通電(對應解鎖開門行為)
2. 再次輸入數字"1", 進行遊戲。
3. 玩家須對著Pi camera進行猜拳，而電腦也會由三種手勢中選一作為對應方的出拳，並將結果顯示在螢幕上。
4. 當玩家贏過電腦 1次之後，樹梅派會發出指定，使磁力所失去通電(模擬開門)
5. 若玩家在獲勝前，已經輸給電腦 3次, 則會將先前拍攝的進入者的照片，透過telegram bot上傳到指定群組，告知群組人員有人試圖闖入宿舍。
6. 並將遊戲暫時鎖定(預設為30秒)，鎖定期間玩家無法再次遊玩

<hr/>

### 使用設備
- 軟體
    * Raspberry Pi OS (樹莓派圖形化界面)
    * TensorFlow (在windows上訓練模型)
    * TensorFlow Lite (訓練模型軟體)
    * OpenCV (影像辨別)
    * Telegram Bot (回傳圖像&資料 的 機器人)
 
- 硬體

| 設備名稱 | 數量| 來源              |價格|
| ---- | ---- | ----------------- |--|
|pi|1|MOLi|-
|pi Camera|1|MOLi|-
|磁力鎖|1|蝦皮|320
|杜邦線|8|MOLi|-
|外接銀幕|1|親友贊助|-
|接電夾|2|親友贊助|-

<hr/>

### 模型的應用（train.py -> check.py -> predict.py） 

- 此部分為 Windows 作業系統上訓練模型需要安裝的套件（train.py -> check.py）
    * `pip install tensorflow`
    * `pip install tflite_model_maker`
    * `pip install numpy`
    * `pip3 install argparse`
    * `pip install opencv-python`

![image](https://user-images.githubusercontent.com/56122682/150295443-61f9201d-c8f9-4022-aafc-cfc5ba2d7ed0.jpg)

![image](https://user-images.githubusercontent.com/56122682/150295688-cedd142a-eabd-4c8e-8582-35e808e7894c.png)
  
- train.py

```python=
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
  
```
  
- check.py 

```python=
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
```

- predict.py

```python=
import cv2
import numpy as np
import tflite_runtime.interpreter as Interpreter
#若是要以Windows作業系統開啓
#import tensorflow as tf
#interpreter = tf.lite.Interpreter(model_path='放入自己訓練模型的路徑')# example：(model_path='sps.tflite')
interpreter = Interpreter(model_path='放入自己訓練模型的路徑')# example：(model_path='sps.tflite')
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
```
  
- 此部分為 Raspberry Pi OS 作業系統上執行 predict.py 所需要安裝的套件（predict.py）
    * `apt-get install python3-tflite-runtime`
    * `pip install tflite_model_maker`
    * `pip install numpy`
    * `pip3 install argparse`
    * `pip install opencv-python`

![image](https://user-images.githubusercontent.com/56122682/150295469-1430e62a-8dae-44f1-a26f-fdc5c031a656.jpg)

<hr/>

## 設備安裝 (需特別注意)

![image](https://user-images.githubusercontent.com/94297365/150322025-2f3ae1a9-cb23-4438-ac3e-f069548aa16e.png)

**地線接 Ground**
<br/>
**火線接 GPIO24**

![image](https://user-images.githubusercontent.com/94297365/150322290-568ad3af-7e30-4e94-97bf-76da8e700665.png)

**樹莓派的地線通過接電夾接到磁力鎖的地線上**
<br/>
**樹莓派的火線通過接電夾接到磁力鎖的火線上**
<br/>
**切勿接錯線路，否則樹莓派會燒掉**
<br/>
![image](https://user-images.githubusercontent.com/94297365/150323585-50fd44b6-ea67-417b-aae1-32832731083d.png)


<hr/>

### 安裝流程

登錄樹莓派後要先更新軟體 
- `sudo apt upgrade`

更新完軟體後安裝python3 
- `sudo apt install python3`

安裝 TensorFlow Lite
- `apt-get install python3-tflite-runtime`
- `pip install tflite_model_maker`

安裝 numpy (矩陣，訓練預測模型)
- `pip install numpy`

安裝 argparse (numpy套件)
- `pip install argparse`

安裝 OpenCV
* `pip install opencv-python`

<hr/>

### Telegram Bot 

三次失敗後，相機原先拍的照片會傳到Telegram Bot房間群裡，並隨機配上一段文字~
(有助於更好的辨認到底是誰一直想闖進別人宿舍)

![LSApic](https://user-images.githubusercontent.com/94297365/150220417-fd63c4bd-9e52-457b-9765-8045b2cd6712.png)

<hr/>

### 參考資料

#### TensorFlow

- Installtion TensorFlow Lite for Python and Run an inference using tflite_runtime https://www.tensorflow.org/lite/guide/python
- TFLite models https://thinkmobile.dev/testing-tensorflow-lite-image-classification-model/
- Customize Post-training quantization on the TensorFLow Lite model https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_image_classification.ipynb#scrollTo=Safo0e40wKZW
- Custom Image Classification Model Using TensorFlow Lite Model Maker https://towardsdev.com/custom-image-classification-model-using-tensorflow-lite-model-maker-68ee4514cd45

#### Telegram Bot

- Create a Telegram Bot https://sendpulse.com/knowledge-base/chatbot/create-telegram-chatbot
- Telegram Bot gets the group chat id https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id
- Telegram Send Photo to Channel (using python-telegram-bot) https://code.luasoftware.com/tutorials/telegram/telegram-send-photo-to-channel/

### 特別感謝

#### 李漢偉 助教大大
- 題材發想
- 技術指導
- 審題

#### 鐘明智 友情協助
- 指導 TensorFlow 模型訓練
- Debug

#### 陳世效 友情協助
- 協助安裝磁力鎖
- 提供電子設備
- 協助排除電路/板相關問題
