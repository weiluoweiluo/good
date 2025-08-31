import os
import pathlib
from PIL import Image
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from nets.arcface import Arcface as arcface
from utils.utils import preprocess_input, resize_image

class ArcfaceFeatureExtractor:
    def __init__(self, model_path="model_data/arcface_mobilenet_v1.pth", backbone="mobilenetv1", input_shape=[112, 112, 3], cuda=False):
        self.model_path = model_path
        self.backbone = backbone
        self.input_shape = input_shape
        self.cuda = cuda
        
        # 初始化模型
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        self.net = arcface(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print(f'{self.model_path} model loaded.')

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
    
    def extract_features(self, image):
        """提取单张图片的人脸特征"""
        with torch.no_grad():
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
            photo = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0))
            
            if self.cuda:
                photo = photo.cuda()
                
            features = self.net(photo).cpu().numpy()
            return features

class FaceRegister:
    def __init__(self, csv_file="face_features.csv"):
        self.csv_file = csv_file
        self.face_label_column = "Label"
        self.face_features_column = "Features"
    
    def register_face(self, image_path, label=None, extractor=None):
        """注册单张人脸"""
        if extractor is None:
            extractor = ArcfaceFeatureExtractor()
            
        if label is None:
            label = pathlib.Path(image_path).stem
            
        try:
            image = Image.open(image_path)
            features = extractor.extract_features(image)
            
            # 准备新行数据
            new_row = {
                self.face_label_column: label,
                self.face_features_column: ",".join(map(str, features[0]))
            }
            
            # 保存到CSV
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file, encoding="utf-8-sig")
                new_df = pd.DataFrame([new_row], columns=df.columns)
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])
                
            df.to_csv(self.csv_file, index=False, encoding="utf-8-sig")
            return True, f"人脸注册成功，标签为: {label}"
            
        except Exception as e:
            return False, f"人脸注册失败: {str(e)}"
    
    def batch_register(self, image_dir, extractor=None):
        """批量注册人脸"""
        if extractor is None:
            extractor = ArcfaceFeatureExtractor()
            
        feature_groups = []
        success_count = 0
        fail_count = 0
        
        for f in pathlib.Path(image_dir).iterdir():
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    image = Image.open(f)
                    features = extractor.extract_features(image)
                    feature_groups.append({
                        self.face_label_column: f.stem,
                        self.face_features_column: ",".join(map(str, features[0]))
                    })
                    success_count += 1
                except Exception as e:
                    print(f"处理文件 {f.name} 失败: {str(e)}")
                    fail_count += 1
        
        if feature_groups:
            # 保存到CSV
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file, encoding="utf-8-sig")
                new_df = pd.DataFrame(feature_groups, columns=df.columns)
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.DataFrame(feature_groups)
                
            df.to_csv(self.csv_file, index=False, encoding="utf-8-sig")
        
        return success_count, fail_count

if __name__ == "__main__":
    # 使用示例
    extractor = ArcfaceFeatureExtractor()
    register = FaceRegister()
    
    while True:
        print("\n请选择操作:")
        print("1. 注册单张人脸")
        print("2. 批量注册人脸")
        print("3. 退出")
        
        choice = input("请输入选项(1/2/3): ")
        
        if choice == "1":
            image_path = input("请输入人脸图片路径: ")
            label = input("请输入人脸标签(留空则使用文件名): ") or None
            success, message = register.register_face(image_path, label, extractor)
            print(message)
            
        elif choice == "2":
            dir_path = input("请输入包含人脸图片的文件夹路径: ")
            success, fail = register.batch_register(dir_path, extractor)
            print(f"批量注册完成: 成功 {success} 个, 失败 {fail} 个")
            
        elif choice == "3":
            print("退出程序")
            break
            
        else:
            print("无效的选项，请重新输入")