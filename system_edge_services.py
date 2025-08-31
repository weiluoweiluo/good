# /usr/bin/env/ python
# -*- coding: utf-8 -*-

import os
import sys
import time
import signal
import argparse
import subprocess
from functools import partial
import multiprocessing
import threading

#可以配置的变量

encryptKEY = '00112233445566778899aabbccddeeff'
origFILE = 'face_features.csv'
encryptFILE = 'face_features.enc'

stop_event = threading.Event()


# 主要菜单
def show_menu():
    print("\n=== 人脸识别系统主菜单 ===")
    print("1. 上传人脸图片进行人脸注册")
    print("2. 通过摄像头截取人脸注册")
    print("3. 上传图片进行人脸库对比")
    print("4. 通过摄像头实时人脸识别")
    print("5. 退出系统")
    choice = input("请输入选项数字: ").strip()
    return choice

def run_module(module_name):
    """统一执行子模块"""
    subprocess.run([sys.executable, module_name])

def setup_args():
    """创建统一的参数配置"""
    args = argparse.Namespace(
        arcface_model_path='model_data/arcface_mobilenet_v1.pth',
        csv_file='face_features.csv',
        save_dir='jpg',
        mtcnn_model_dir='infer_models'
    )
    return args

def single_image_register():
    from Image_register import face_register_single_image
    face_register_single_image()

def realtime_register():
    from realtime_register import face_register
    face_register()

def image_recognition():
    from Image_recognition import recognize
    args = setup_args()
    recognize(args)

def realtime_recognition():
    from realtime_recognition import realtime_recognize
    args = setup_args()
    realtime_recognize(args)

def encrypt(stop):
    while not stop.is_set():
        if(os.path.exists(encryptFILE)):
            ret = os.system(f'python sm4_decrypt.py {encryptFILE} {origFILE} -k {encryptKEY}')
            if(ret==0):
                print('文件解密成功')
                os.remove(encryptFILE)
        time.sleep(1)

    
def MQTTReceiver(stop):
    process3 = subprocess.Popen(['python', 'MQTTReceiver.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while not stop.is_set():
        time.sleep(1)
    process3.send_signal(signal.SIGTERM)
    #等待进程结束
    process3.wait()



if __name__ == "__main__":
    #先启动SM4解密处理、MQTT客户端、的两个线程
    t1 = threading.Thread(target=encrypt,args=(stop_event,))
    t1.start()
    t2 = threading.Thread(target=MQTTReceiver, args=(stop_event,))
    t2.start()

    menu_actions = {
        '1': single_image_register,
        '2': realtime_register,
        '3': image_recognition,
        '4': realtime_recognition
    }

    while True:
        choice = show_menu()
        if choice == '5':
            stop_event.set()
            t1.join()
            t2.join()
            print("\n感谢使用，再见！")
            break
            
        action = menu_actions.get(choice)
        if action:
            try:
                action()
                input("\n操作完成，按回车返回主菜单...")
            except Exception as e:
                print(f"发生错误: {str(e)}")
        else:
            print("无效的输入，请重新选择")