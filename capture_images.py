import cv2
import os
import time

def capture_images(output_dir="jpg"):
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开摄像头
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            print(f'当前cam序列号是{str(i)}')
            break
        except:
            time.sleep(0.2)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 's' 保存图像，按 'q' 退出")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存图像
            img_name = f"image_{count}.jpg"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"图像已保存到: {img_path}")

            # 手动输入标签
            # label = input('请输入图片标签')
            label = input('请输入图片标签: ')
            if(label.strip() == ""):
                print("标签不能为空，请重新输入")
                continue

            # 将标签保存到文件名中（可选）
            labeled_img_name = f"{label}_{count}.jpg"
            labeled_img_path = os.path.join(output_dir, labeled_img_name)
            if os.path.exists(labeled_img_path):
                # 如果文件已存在，添加计数器
                counter = 0
                while os.path.exists(labeled_img_path):
                    labeled_img_name = f"{label}_{count}_{counter}.jpg"
                    labeled_img_path = os.path.join(output_dir, labeled_img_name)
                    counter += 1
            os.rename(img_path, labeled_img_path)
            print(f"图像已重命名为: {labeled_img_path}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()

    # 使用示例
if __name__ == "__main__":
    capture_images()