import os
import cv2
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from Mtcnn_process import infer_image, draw_face
from featur_extraction import ArcfaceFeatureExtractor

def recognize(args):
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

    # 持续识别循环
    while True:
        # 获取输入路径
        image_path = input('\n请输入待识别图片路径（输入q退出）: ').strip()
        if image_path.lower() == 'q':
            break

        # 读取并验证图片
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {image_path}")
        except Exception as e:
            print(f"错误：{e}")
            continue

        # 检测人脸
        output_dir = 'test'
        os.makedirs(output_dir, exist_ok=True)
        boxes_c, landmarks = infer_image(img)  # 获取landmarks
        if boxes_c is not None and len(boxes_c) > 0:
            # 获取第一个检测到的人脸
            box = boxes_c[0]
            x1, y1, x2, y2 = map(int, box[:4])  # 确保只取坐标
            face = img[y1:y2, x1:x2]

            # 裁剪并保存人脸
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
            threshold = 0.6  # 根据实际测试调整阈值
            if max_sim > threshold:
                print(f"\033[92m识别成功！{best_match} (相似度: {float(max_sim):.4f})\033[0m")
                # 显示检测结果
                result_img = draw_face(img, boxes_c, landmarks)
                cv2.imshow('识别结果', result_img)
                cv2.waitKey(3000)  # 显示3秒
                cv2.destroyAllWindows()
            else:
                print(f"\033[91m识别失败：最高相似度 {float(max_sim):.4f} 低于阈值\033[0m")
        else:
            print("未检测到人脸，请检查图片质量（可能原因：低分辨率/侧脸/遮挡）")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人脸识别系统")
    parser.add_argument('--arcface_model_path', type=str, default='model_data/arcface_mobilenet_v1.pth',
                       help='ArcFace模型路径')
    parser.add_argument('--csv_file', type=str, default='face_features.csv',
                       help='特征数据库CSV文件路径')
    args = parser.parse_args()

    print("===== 进入人脸识别模式 =====")
    recognize(args)