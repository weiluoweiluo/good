# /usr/bin/env/ python
# -*- coding: utf-8 -*-
import os
import re
import time
import sys
import signal
import urllib
import operator
import binascii
import threading
import paho.mqtt.client as mqtt


#全局变量、读取的时候可以直接使用、修改则需要函数体声明，或者采用globals()函数来操作全局变量
MQTTHOST = "192.168.3.217"
MQTTPORT = 1883
USERNAME = "root"
PASSWORD = "passwd"
CLIENTID = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
HEARTBEAT = 60
client = mqtt.Client(callback_api_version = mqtt.CallbackAPIVersion.VERSION1,client_id=CLIENTID)
TOPIC = 'phytium/face'
MSG=""
sessionID = ''
Num = 0
dictFile = {}

#联合跑测的时候featurePath配置为空即可
featurePath = ''

encryptKEY = '00112233445566778899aabbccddeeff'
origFILE = 'face_features.csv'
encryptFILE = 'face_features.enc'

def on_connect(client, userdata, flags, rc):
    if(rc==0):
        print(f"接收端连接Broker成功\n订阅主题 -> {TOPIC}\n")
        client.subscribe(TOPIC)
    else:
        print(f"接收端连接Broker失败，失败代码 {str(rc)}\n")

# 定义回调函数
def on_message(client, userdata, msg):
    global sessionID
    global Num
    global dictFile
    MQTT_Rx_Buff = str(msg.payload, encoding="utf-8")
    if (MQTT_Rx_Buff.startswith('file-start') or MQTT_Rx_Buff.startswith("file-body") or MQTT_Rx_Buff.startswith("file-end")):
        if(sessionID ==''):
            sessionID = MQTT_Rx_Buff.split(',')[1]
        if(dictFile == {}):
            dictFile[sessionID] = []
        if(Num==0):
            Num = int(MQTT_Rx_Buff.split(',')[3])
        on_file_message(sessionID,Num,msg)
    else:
        print(f"消息\"{MQTT_Rx_Buff}\"接受成功，等待新的消息推送\n")


def on_file_message(sID,count,msg):
    msgList =  str(msg.payload).split(',')
    if(sID == msgList[1]):
        #全局变量修改
        globals()['dictFile'][sID].append(msgList) 
        glen = len(globals()['dictFile'][sID])
        print(f'当前长度{glen},需要长度{count}\n')
        if(glen==count):
            msgPartList = globals()['dictFile'][sID]
            # newMsgPartList = sorted(msgPartList, key=operator.itemgetter(2), reverse=False)
            newMsgPartList = sorted(msgPartList, key=lambda sortkey:int(sortkey[2]), reverse=False)
            filePartList = list(map(lambda row: row[4], newMsgPartList))
            filePartLen = len(filePartList)
            received_hex = b''

            newfilename = filePartList[0][:-1]
            with open(featurePath + newfilename, 'wb') as f:
                for i in range(1,filePartLen-1):
                    received_hex += bytes.fromhex(filePartList[i][:-1])
                    if(int(i/(filePartLen-1)*100) % 10 == 0 ):
                        print("文件生成进度{:.2%}\n".format(i/(filePartLen-1)))
                f.write(received_hex)
                print("文件生成进度{:.2%}\n".format(1))
                f.close()
            
            print(f'文件{newfilename}生成成功，等待新的消息推送\n')
            globals()['dictFile'] = {}
            globals()['sessionID'] = ''
            globals()['Num'] = 0


def handle_signal():
    client.loop_stop()
    client.disconnect()
    print("MQTT接收客户端退出")
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGTERM,  handle_signal)
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(USERNAME, PASSWORD)
    while(True):
        try:    
            errcode =  client.connect(MQTTHOST, MQTTPORT, HEARTBEAT)
        except:
            print(f'MQTT服务器连接错误\n')
            errcode = -1
        if(mqtt.MQTT_ERR_SUCCESS == errcode):
            print(f'连接MQTT服务器{MQTTHOST}成功\n')
            break
        else:
            print(f'连接MQTT服务器{MQTTHOST}失败，错误代码为{errcode}\n')
            retrytag = input('是否重连，重连请输入大写的Y，其他输入将退出客户端\n')
            if not(retrytag=="Y"):
                break
    if(mqtt.MQTT_ERR_SUCCESS == errcode):
        client.loop_start()
        while True:
            MSG = input("请输入命令，exit退出\n")
            if(MSG=='exit'):
                client.loop_stop()
                client.disconnect()
                input("信息接收端即将退出，任意键退出程序")
                break