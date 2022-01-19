# 來玩遊戲吧

### 概念發想
- 由於房間內有一個人是由住服組安排進來的，所以他的朋友們經常來找他玩，又一直不把門帶上，所以有時玩遊戲玩到一半叫感到一陣涼意，然後就死掉了。
- 另外他的朋友們還很喜歡半夜三更來找他，然後大半夜的門也沒關緊，時常造成大家隔天早上一瀉千里，非常痛苦。
<hr/>

### 使用設備
- 軟體
    * Raspberry Pi OS (樹莓派圖形化界面)
    * TensorFlow Lite
    * OpenCV
    * Telegram Bot
 
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

### 模型訓練 
- 此部分為window 上訓練模型需要安裝的套件（已有上傳訓練好的模型下載，這部分可省略）
    * `pip install tflite_model_maker`
    * `pip install os`
    * `pip install numpy`
    * `pip install opencv-python`

<hr/>

### 安裝流程
登錄樹莓派後要先更新軟體 
- `sudo apt upgrade`

更新完軟體後安裝python3 
- `sudo apt install python3`

安裝 TensorFlow Lite
- `pip3 install tflite-runtime`
安裝 numpy (矩陣，訓練預測模型)
- `pip3 install numpy`

安裝 argparse (numpy套件)
- `pip3 install argparse`

安裝 OpenCV
* `pip3 install opencv-python`


<hr/>

### Telegram Bot 
