import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= 配置区域 =================

MODEL_PATH = r"C:\Users\Administrator\PycharmProjects\geo_detect\geometry_dataset_final\last_model.pth"


INPUT_IMAGE_PATH = r"C:\Users\Administrator\PycharmProjects\geo_detect\geometry_dataset_final\test.jpg"


OUTPUT_IMAGE_NAME = "result_final.jpg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384
def main():
    # --- 检查路径 ---
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"[Error] 找不到输入图片: {INPUT_IMAGE_PATH}")
        print("请修改 INPUT_IMAGE_PATH")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[Error] 找不到模型权重: {MODEL_PATH}")
        return

    # --- 1. 加载模型 (离线模式) ---
    print(f"正在加载模型...")
    # 结构必须与训练时完全一致
    model = smp.Unet(
        encoder_name="tu-swin_large_patch4_window12_384_in22k",
        encoder_weights=None,  # 禁止联网
        in_channels=3,
        classes=3,
        activation=None
    )

    # 加载权重
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(">>> 模型加载完毕。")
    print(f"正在读取图片: {INPUT_IMAGE_PATH}")
    original_img = cv2.imread(INPUT_IMAGE_PATH)
    if original_img is None:
        print("[Error] 图片读取失败，请检查文件是否损坏。")
        return

    h_orig, w_orig = original_img.shape[:2]
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # 预处理变换 (Resize -> Normalize -> ToTensor)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 增加 batch 维度 [3, H, W] -> [1, 3, H, W] 并送到设备
    input_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)


    print("正在识别")
    with torch.no_grad():
        logits = model(input_tensor)
        # Sigmoid 将输出转换为 0-1 的概率值
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # 结果形状 [3, 384, 384]


    # 将通道移到最后: [384, 384, 3]
    probs = np.transpose(probs, (1, 2, 0))

    # 将预测的 Mask 缩放回原图分辨率
    mask_resized = cv2.resize(probs, (w_orig, h_orig))

    # 准备画布 (复制原图)
    result_viz = original_img.copy()

    # 设定置信度阈值 (大于此值的像素才会被画出来)
    th = 0.5

    # 分离通道数据 (注意这里是模型输出顺序: Line, Circle, Point)
    lines_mask = mask_resized[:, :, 0]
    circles_mask = mask_resized[:, :, 1]
    points_mask = mask_resized[:, :, 2]

    # --- 开始绘制 (使用 OpenCV 的 BGR 颜色空间) ---

    # 1. 画直线 (红色: 0,0,255)
    # 使用实心覆盖，这在低 Loss 下效果最好
    result_viz[lines_mask > th] = [0, 0, 255]

    # 2. 画圆 (绿色: 0,255,0)
    result_viz[circles_mask > th] = [0, 255, 0]

    # 3. 画交点 (蓝色: 255,0,0)
    # 点是热力图，阈值稍微调高一点点，让点看起来更聚拢、更锐利
    result_viz[points_mask > (th + 0.15)] = [255, 0, 0]

    # --- 5. 保存结果 ---
    cv2.imwrite(OUTPUT_IMAGE_NAME, result_viz)

    print(f"处理完成！")
    print(f"结果已保存为:{OUTPUT_IMAGE_NAME}")
    print("请打开图片查看")


if __name__ == "__main__":
    main()