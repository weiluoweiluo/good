import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
import pdb


from utils1.utils import generate_bbox, py_nms, convert_to_square
from utils1.utils import pad, calibrate_box, processed_image

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='infer_models',      help='PNet、RNet、ONet三个模型文件存在的文件夹路径')
args = parser.parse_args()

#device = torch.device("cuda")
device = torch.device("cpu")

# 获取P模型
pnet = torch.jit.load(os.path.join(args.model_path, 'PNet.pth'))
pnet.to(device)
softmax_p = torch.nn.Softmax(dim=0)
pnet.eval()

# 获取R模型
rnet = torch.jit.load(os.path.join(args.model_path, 'RNet.pth'))
rnet.to(device)
softmax_r = torch.nn.Softmax(dim=-1)
rnet.eval()

# 获取O模型
onet = torch.jit.load(os.path.join(args.model_path, 'ONet.pth'))
onet.to(device)
softmax_o = torch.nn.Softmax(dim=-1)
onet.eval()


# 使用PNet模型预测
def predict_pnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    # 执行预测
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用RNet模型预测
def predict_rnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用ONet模型预测
def predict_onet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, landmark_pred = onet(infer_data)
    cls_prob = softmax_o(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()


# 获取PNet网络输出结果
def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    """
    net_size = 12
    # 人脸和输入图像的比率
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    # 图像金字塔
    while min(current_height, current_width) > net_size:
        # 类别和box
        cls_cls_map, reg = predict_pnet(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor  # 继续缩小图像做金字塔
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape

        if boxes.size == 0:
            continue
        # 非极大值抑制留下重复低的box
        keep = py_nms(boxes[:, :5], 0.5, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    # 将金字塔之后的box也进行非极大值抑制
    keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
    all_boxes = all_boxes[keep]
    # box的长宽
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    # 对应原图的box坐标和分数
    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])
    boxes_c = boxes_c.T

    return boxes_c


# 获取RNet网络输出结果
def detect_rnet(im, dets, thresh):
    """通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
    """
    h, w, c = im.shape
    # 将pnet的box变成包含它的正方形，可以避免信息损失
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    # 调整超出图像的box
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
    for i in range(int(num_boxes)):
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        except:
            continue
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None

    keep = py_nms(boxes, 0.6, mode='Union')
    boxes = boxes[keep]
    # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes_c


# 获取ONet模型预测结果
def detect_onet(im, dets, thresh):
    """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    num_boxes = dets.shape[0]
    cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) / 128
        cropped_ims[i, :, :, :] = img

    cls_scores, reg, landmark = predict_onet(cropped_ims)

    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        landmark = landmark[keep_inds]
    else:
        return None, None

    w = boxes[:, 2] - boxes[:, 0] + 1

    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
    landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
    boxes_c = calibrate_box(boxes, reg)

    keep = py_nms(boxes_c, 0.6, mode='Minimum')
    boxes_c = boxes_c[keep]
    landmark = landmark[keep]
    return boxes_c, landmark


# 预测图片
def infer_image(im):
    # 调用第一个模型预测
    boxes_c = detect_pnet(im, 20, 0.79, 0.9)
    if boxes_c is None:
        return None, None
    # 调用第二个模型预测
    boxes_c = detect_rnet(im, boxes_c, 0.6)
    if boxes_c is None:
        return None, None
    # 调用第三个模型预测
    boxes_c, landmark = detect_onet(im, boxes_c, 0.7)
    if boxes_c is None:
        return None, None

    return boxes_c, landmark


# 画出人脸框和关键点
def draw_face(img, boxes_c, landmarks):
    if boxes_c is None or landmarks is None:
        return img  # 如果没有检测到人脸或关键点，直接返回原图
    
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # 画人脸框
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        
        # 判别为人脸的置信度
        cv2.putText(img, '{:.2f}'.format(score),
                    (corpbbox[0], corpbbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 画关键点
    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i]) // 2):
                cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
    
    return img
    


if __name__ == '__main__':
    
    # 处理后数据集路径
    output_dir = 'mtcnn_result'
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 原始数据集文件路径
    file_path = 'cls_train.txt'
    
    # 读取文件中的图片路径
    with open(file_path, 'r', encoding='utf-8') as f:
        image_paths = f.readlines()  # 读取所有行
    
    # 遍历每个图片路径
    for line_num, image_path in enumerate(image_paths):
        # 提取图片的实际路径（根据您的文件格式调整）
        image_path = image_paths[line_num].split(';')[1].split()[0]
        
        if os.path.exists(image_path):
            # 加载图像
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图像，请检查文件路径或文件格式: {image_path}")
                continue  # 跳过当前循环
            
            # 人脸检测，预测图片获取人脸的box和关键点
            boxes_c, landmarks = infer_image(img)
            print(boxes_c)
            print(landmarks)
            
            # 如果检测到人脸，取第一个检测到的人脸
            if boxes_c is not None and len(boxes_c) > 0:
                box = boxes_c[0]
                x1, y1, x2, y2, score = map(int, box)  # 获取坐标和分数
                
                # 裁剪人脸区域
                face = img[y1:y2, x1:x2]
                
                if face is None or face.size == 0:
                    print("无法读取裁剪后的人脸图像，或图像数据为空！")
                    # 根据需要删除原始图像
                    # os.remove(image_path)
                else:
                    # 获取原始图像的文件名和扩展名
                    _, filename = os.path.split(image_path)
                    filename_without_ext, ext = os.path.splitext(filename)
                    
                    # 构建新的保存路径
                    new_image_name = f"cropped_{filename_without_ext}{ext}"
                    save_path = os.path.join(output_dir, new_image_name)
                    
                    face_resized = cv2.resize(face, (96, 112), interpolation=cv2.INTER_LINEAR)
                    # 保存裁剪后的人脸图像
                    result = cv2.imwrite(save_path, face_resized)
                    
                    if result:
                        print(f"裁剪后的人脸图像已保存到: {save_path}")
                    else:
                        print(f"保存裁剪后的人脸图像失败，请检查写入路径和权限: {save_path}")
            else:
                print(f'图像 {image_path} 中未检测到人脸，将删除原始图像。')
                os.remove(image_path)
        else:
            print(f"图像路径不存在: {image_path}")
    
    print("人脸检测与裁剪完成")