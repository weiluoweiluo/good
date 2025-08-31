# /usr/bin/env/ python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from Mtcnn_process import infer_image
from featur_extraction import ArcfaceFeatureExtractor

def realtime_recognize(args):
    # 初始化特征提取器
    extractor = ArcfaceFeatureExtractor(
        model_path=args.arcface_model_path,
        backbone="mobilenetv1",
        input_shape=[112, 112, 3],
        cuda=False
    )

    # 加载特征数据库
    if not os.path.exists(args.csv_file):
        print(f"错误：特征数据库 {args.csv_file} 不存在，请先进行注册！")
        return
    df = pd.read_csv(args.csv_file, encoding="utf-8-sig")

    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 使用默认摄像头（索引0）
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    print("===== 开始实时人脸识别 =====")
    frame_count = 0
    process_interval = 30  # 每30帧处理一次
    display_duration = 20  # 识别结果持续显示的帧数
    last_result = None  # 缓存上一次的识别结果

    # 创建显示窗口
    cv2.namedWindow('实时人脸识别', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('实时人脸识别', 640, 480)

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头图像！")
            break

        # 每process_interval帧处理一次
        if frame_count % process_interval == 0:
            # 检测人脸
            boxes_c, landmarks = infer_image(frame)
            if boxes_c is not None and len(boxes_c) > 0:
                # 获取第一个检测到的人脸
                box = boxes_c[0]
                # 确保坐标是整数且在图像范围内
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(frame.shape[1], int(box[2]))
                y2 = min(frame.shape[0], int(box[3]))
                
                # 检查裁剪区域是否有效
                if x2 > x1 and y2 > y1:
                    face = frame[y1:y2, x1:x2]
                    
                    try:
                        # 调整人脸大小
                        face_img = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)
                        
                        # ArcFace特征提取
                        face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        query_feat = extractor.extract_features(face_img_pil)
                        query_feat /= np.linalg.norm(query_feat)  # L2归一化

                        # 数据库比对
                        max_sim = -1
                        best_match = None
                        for _, row in df.iterrows():
                            # 解析数据库特征
                            db_feat = np.fromstring(row['Features'], sep=',', dtype=np.float32)
                            db_feat /= np.linalg.norm(db_feat)
                            
                            # 计算余弦相似度
                            similarity = np.dot(query_feat.flatten(), db_feat.flatten())
                            
                            if similarity > max_sim:
                                max_sim = similarity
                                best_match = row['Label']

                        # 显示结果
                        threshold = 0.7  # 根据实际测试调整阈值
                        if max_sim > threshold:
                            print(f"\033[92m识别成功！{best_match} (相似度: {float(max_sim):.4f})\033[0m")
                            last_result = {
                                "box": (x1, y1, x2, y2),
                                "label": best_match,
                                "color": (0, 255, 0)  # 绿色框
                            }
                        else:
                            print(f"\033[91m识别失败：最高相似度 {float(max_sim):.4f} 低于阈值\033[0m")
                            last_result = {
                                "box": (x1, y1, x2, y2),
                                "label": "Unknown",
                                "color": (0, 0, 255)  # 红色框
                            }
                    except Exception as e:
                        print(f"人脸处理错误: {str(e)}")
                        last_result = None
                else:
                    print("无效的人脸区域坐标")
                    last_result = None
            else:
                print("未检测到人脸")
                last_result = None

        # 在当前帧上显示上一次的识别结果
        if last_result:
            x1, y1, x2, y2 = last_result["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), last_result["color"], 2)
            cv2.putText(frame, last_result["label"], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, last_result["color"], 2)

        # 调整显示图像的大小
        display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        # 显示结果图像
        cv2.imshow('实时人脸识别', display_frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % display_duration == 0:
            last_result = None  # 清空缓存结果

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实时人脸识别系统")
    parser.add_argument('--arcface_model_path', type=str, default='model_data/arcface_mobilenet_v1.pth',
                       help='ArcFace模型路径')
    parser.add_argument('--csv_file', type=str, default='face_features.csv',
                       help='特征数据库CSV文件路径')
    args = parser.parse_args()

    print("===== 进入实时人脸识别模式 =====")
    realtime_recognize(args)
