# 來玩遊戲吧


### 概念發想
- 由於房間內有一個人是由住服組安排進來的，所以他的朋友們經常來找他玩，又一直不把門帶上，所以有時玩遊戲玩到一半叫感到一陣涼意，然後就死掉了。
- 另外他的朋友們還很喜歡半夜三更來找他，然後大半夜的門也沒關緊，時常造成大家隔天早上一瀉千里，非常痛苦。
- **本遊戲為石頭剪刀佈**

<hr/>

### 使用設備
- 軟體
    * Raspberry Pi OS (樹莓派圖形化界面)
    * TensorFlow Lite (訓練模型軟體)
    * OpenCV (影像辨別)
    * Telegram Bot (回傳圖像&資料 的 機器人)
 
- 硬體

| 設備名稱 | 數量| 來源              |價格|
| ---- | ---- | ----------------- |--|
|pi|1|MOLi|-
|pi Camera|1|MOLi|-
|磁力鎖|1|蝦皮|320
|5V繼電器|1|MOLi|-
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
  
- 此部分為 Raspberry Pi OS 作業系統上執行 predict.py 所需要安裝的套件（predict.py）
    * `apt-get install python3-tflite-runtime`
    * `pip install tflite_model_maker`
    * `pip install numpy`
    * `pip3 install argparse`
    * `pip install opencv-python`

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
