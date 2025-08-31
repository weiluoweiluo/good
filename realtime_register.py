import os
import cv2
import argparse
from PIL import Image
import pandas as pd
import numpy as np
import torch
from capture_images import capture_images
from Mtcnn_process import infer_image, draw_face
from featur_extraction import ArcfaceFeatureExtractor, FaceRegister

def face_register():
    parser = argparse.ArgumentParser(description="人脸采集与注册流程")
    parser.add_argument('--save_dir', type=str, default='jpg', help='保存采集图像的目录')
    parser.add_argument('--mtcnn_model_dir', type=str, default='infer_models', help='MTCNN模型目录')
    parser.add_argument('--arcface_model_path', type=str, default='model_data/arcface_mobilenet_v1.pth', help='ArcFace模型路径')
    parser.add_argument('--csv_file', type=str, default='face_features.csv', help='特征注册CSV文件路径')
    args = parser.parse_args()

    # 1. 图像采集
    print("开始图像采集...")
    # 仅保留摄像头采集逻辑
    capture_images(output_dir=args.save_dir)
    print("图像采集完成。")

    # 2. MTCNN人脸检测与裁剪
    print("开始人脸检测与裁剪...")
    output_dir = 'mtcnn_result'
    os.makedirs(output_dir, exist_ok=True)

    # 遍历采集的图像
    for img_file in os.listdir(args.save_dir):
        img_path = os.path.join(args.save_dir, img_file)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        # 检测人脸
        boxes_c, landmarks = infer_image(img)  # 修改这里，获取landmarks
        if boxes_c is not None and len(boxes_c) > 0:
            # 获取第一个检测到的人脸
            box = boxes_c[0]
            x1, y1, x2, y2 = map(int, box[:4])  # 修改这里，确保只取坐标
            face = img[y1:y2, x1:x2]

            # 裁剪并保存人脸
            face_resized = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)
            # 使用新的文件名避免冲突
            base_name, ext = os.path.splitext(img_file)
            save_path = os.path.join(output_dir, img_file)
            if os.path.exists(save_path):
                # 如果文件已存在，添加计数器
                count = 0
                while os.path.exists(save_path):
                    save_path = os.path.join(output_dir, f"{base_name}_{count}{ext}")
                    count += 1
            cv2.imwrite(save_path, face_resized)
            print(f"人脸裁剪完成，保存到: {save_path}")
        else:
            print(f"未检测到人脸，跳过: {img_path}（可能原因：低分辨率/侧脸/遮挡）")
            
    print("人脸检测与裁剪完成。")

    # 3. ArcFace特征提取与注册
    print("开始特征提取与注册...")
    extractor = ArcfaceFeatureExtractor(
        model_path=args.arcface_model_path,
        backbone="mobilenetv1",
        input_shape=[112, 112, 3],
        cuda=False
    )
    register = FaceRegister(csv_file=args.csv_file)

    # 批量注册裁剪后的人脸
    success, fail = register.batch_register(output_dir, extractor)
    print(f"特征提取与注册完成: 成功 {success} 个, 失败 {fail} 个")

if __name__ == "__main__":
    face_register()