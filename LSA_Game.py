import picamera
import RPi.GPIO as GPIO
import time
import random
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import telegram
from datetime import datetime
import telepot
from telepot.loop import MessageLoop
from pprint import pprint

GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.OUT)

def DoorControl() :  #control the door to open||lock
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(24, GPIO.OUT)
    DoorLock()
    x = int(input())
    if x == 1 :
        DoorOpen()

def DoorLock() :  #lock the door
    GPIO.output(24, GPIO.HIGH)

def DoorOpen() :  #open the door
    GPIO.setwarnings(False)  #忽視警告
    GPIO.cleanup()
    print("Welcome")

def Camera(camera) :
    #camera = picamera.PiCamera()
    #camera.resolution = (500, 500)
    camera.start_preview()
    time.sleep(5)
    camera.capture('/home/pi/Desktop/text1.jpg')
    camera.stop_preview()
    Player = str(SSP())
    return Player

camera = picamera.PiCamera()
camera.resolution = (1000, 1000)
camera.framerate = 30
camera.rotation = 180

def Play() :
    Lose = 0
    Round = 1
    
    #camera.vflip = True
    
    while Lose < 1 :
        now = datetime.now().time()
        print("now: ", now)
        print("Round " + str(Round))
        
        Use = ["Scissors","Stone","Paper"]
        random.shuffle(Use)
        Computer = Use[0]  #computer use
        Player = Camera(camera)
        NowWinner = Winner(Computer, Player)
        
        print("Computer: " + Computer)
        print("Player: " + Player)
        print("---------------------")
        if NowWinner == 0 :
            print("Winner: none")
        else :
            print("Winner: " + NowWinner)
        print()
        Round += 1
        
        if NowWinner == "Computer" :
            Lose += 1
            print("Now Lose " + str(Lose))
            time.sleep(5)
        elif NowWinner == 0 :
            print("Now Lose " + str(Lose))
            time.sleep(5)
        elif NowWinner == "Player" :
            DoorOpen()
            time.sleep(5)
            break
    if Lose == 1 :
        print("You Lose")
        TBT(now)
        for x in range(30) :
            print("Door Clossing: " + str(30-x))
            time.sleep(1)
            
def Winner(Computer, Player) :
    Use = ["Scissors","Stone","Paper", "Scissors"]
    if Computer == Player :
        return 0
    else :
        for x in range(len(Use)) :
            if Computer == Use[x] :
                if Player == Use[x+1] :
                    return "Player"
                else :
                    return "Computer"
        
def SSP() :
    interpreter = Interpreter(model_path='sps3.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image_path='text1.jpg'
    img = cv2.imread(image_path)
    img = cv2.resize(img,(224,224))#check.py 顯示的大小（長寬）
    #Preprocess the image to required size and cast
    input_shape = input_details[0]['shape']
    input_tensor= np.array(np.expand_dims(img,0))

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    #print(pred)
    class_ind = {0:"Paper",1:"Scissors",2:"Stone"}
    highest_pred_loc = np.argmax(pred)#最大值就是預測結果
    hand_name = class_ind[highest_pred_loc]
    return hand_name

def Face() :
    camera.resolution = (300, 300)
    camera.start_preview()
    time.sleep(5)
    camera.capture('/home/pi/Desktop/face1.jpg')
    camera.stop_preview()

def TBT(now):
    #print("now: ", now)
    #text = "time: " + str(now)
    #bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    TELEGRAM_BOT_TOKEN = '5073450987:AAH8WSzx1ju52rWp0Z1B3phPWZqeLCOkPyw'
    # https://api.telegram.org/"替換成bot-token"/getUpdates
    # 在裡面找到群組的chat id
    TELEGRAM_CHAT_ID = '-625214662'
    # 放入指定的圖片位址
    PHOTO_PATH = r'/home/pi/Desktop/face1.jpg'
    # 宣告一個telegram_bot的(放入bot_token)
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    rand = random.randint(1,3)
    if (rand == 1):
        # 想輸入的訊息
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="這個人想闖進宿舍(指 ")
    elif(rand == 2):
        # 想輸入的訊息
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="可黏阿，下次請再接再厲QQ")
    else:
        # 想輸入的訊息
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="警察先生~這裡有怪人出沒!")
    # 想傳入的照片位址
    bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=open(PHOTO_PATH, 'rb'))

def InsertPassword() :  #error no use
    TELEGRAM_BOT_TOKEN = '5073450987:AAH8WSzx1ju52rWp0Z1B3phPWZqeLCOkPyw'
    print("yes1")
    # https://api.telegram.org/"替換成bot-token"/getUpdates
    # 在裡面找到群組的chat id
    TELEGRAM_CHAT_ID = '-625214662'
    PHOTO_PATH = r'/home/pi/Desktop/face1.jpg'
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=open(PHOTO_PATH, 'rb'))
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="someone is comming, 'yes' or 'no'")
    print("yes2")
    telepot.Bot("key").message_loop(command)
    
def command(msg) :  #no use
    pprint("yes3")
    Ans = msg['text']
    pprint(Ans)
    time.sleep(10)
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    if Ans == "/yes" :
        DoorOpen()
    elif Ans == "/no" :
        print("no getout")
    else :
        print("admin no see... ...")
    
def main() :
    DoorLock()
    #InsertPassword()
    ok = 0
    while ok == 0 :
        GiveFace = int(input("Press 1 to Give me your face:"))
        if GiveFace == 7414 :
            DoorOpen()
            ok = 1
        elif GiveFace == 1 :
            Face()
            PlayNow = int(input("Press 1 to play:"))
            if PlayNow == 1 :
                Play()
                ok = 1
        #elif PlayNow == 2 :
            #InsertPassword()
    #DoorControl()
        else :
            print("command error, please try again")
    
if __name__ == "__main__" :
    main()