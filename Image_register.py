import os
import cv2
from PIL import Image
from Mtcnn_process import infer_image
from featur_extraction import ArcfaceFeatureExtractor, FaceRegister

def face_register_single_image():
    # 1. 输入图片路径
    while True:
        image_path = input("请输入要注册的人脸图片路径: ").strip()
        if os.path.exists(image_path):
            break
        print(f"错误：图片路径不存在 - {image_path}，请重新输入")

    # 2. 配置参数
    config = {
        'mtcnn_model_dir': 'infer_models',
        'arcface_model_path': 'model_data/arcface_mobilenet_v1.pth',
        'csv_file': 'face_features.csv',
        'temp_dir': 'temp_faces'
    }

    # 3. 创建临时目录
    os.makedirs(config['temp_dir'], exist_ok=True)

    # 4. MTCNN人脸检测与裁剪
    print("\n开始人脸检测与裁剪...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return False

    boxes_c, landmarks = infer_image(img)
    if boxes_c is None or len(boxes_c) == 0:
        print(f"未检测到人脸，注册失败: {image_path}（可能原因：低分辨率/侧脸/遮挡）")
        return False
    
    # 获取第一个检测到的人脸
    box = boxes_c[0]
    x1, y1, x2, y2 = map(int, box[:4])
    face = img[y1:y2, x1:x2]

    # 裁剪并保存人脸到临时文件
    face_resized = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)
    temp_face_path = os.path.join(config['temp_dir'], "temp_face.jpg")
    cv2.imwrite(temp_face_path, face_resized)
    print(f"人脸裁剪完成，临时保存到: {temp_face_path}")

    # 5. 输入注册标签
    while True:
        name = input("\n请输入该人脸的注册名称(标签): ").strip()
        if name:  # 确保名称不为空
            break
        print("错误：注册名称不能为空，请重新输入")

    # 6. ArcFace特征提取与注册
    print("\n开始特征提取与注册...")
    extractor = ArcfaceFeatureExtractor(
        model_path=config['arcface_model_path'],
        backbone="mobilenetv1",
        input_shape=[112, 112, 3],
        cuda=False
    )
    register = FaceRegister(csv_file=config['csv_file'])

    # 注册单张人脸 - 调用 register_face 
    success, message = register.register_face(temp_face_path, name, extractor)
    
    # 清理临时文件
    if os.path.exists(temp_face_path):
        os.remove(temp_face_path)

    # 7. 显示结果
    print(message)  # 打印注册结果消息
    if success:
        print(f"\n注册成功！\n图片路径: {image_path}\n注册名称: {name}")
        return True
    else:
        print("\n注册失败！")
        return False

if __name__ == "__main__":
    print("=== 单张人脸注册程序 ===")
    result = face_register_single_image()
    if result:
        exit(0)
    else:
        exit(1)