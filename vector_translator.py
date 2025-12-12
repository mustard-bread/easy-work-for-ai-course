import cv2
import json
import numpy as np
import os

# ================= 配置区域 =================
CONFIG = {
    # 1. 输入的 JSON 文件路径 (上一步生成的)
    "input_json_path": "final_vectors.json",

    # 2. 原始图像路径 (用于提供画布尺寸和背景参照)
    # 如果只想看纯矢量在黑背景上，可以将这里设为 None
    # "input_image_path": None,
    "input_image_path": r"C:\Users\Administrator\PycharmProjects\geo_detect\result_final.jpg",

    # 3. 输出的可视化图像路径
    "output_viz_path": "visualized_vectors.jpg",

    # 4. 绘图样式配置 (BGR 颜色 和 线宽)
    "style": {
        # 蓝色点，半径5，实心
        "point_color": (255, 0, 0),
        "point_radius": 6,

        # 绿色圆，线宽2
        "circle_color": (0, 255, 0),
        "circle_thickness": 2,

        # 红色线，线宽3
        "line_color": (0, 0, 255),
        "line_thickness": 3,

        # 背景透明度 (0.0为原图，1.0为纯白，0.7表示原图变淡70%)
        "background_fade": 0.7
    }
}


# ===========================================

def visualize_vectors(config):
    json_path = config['input_json_path']
    img_path = config["input_image_path"]
    style = config['style']

    # 1. 加载 JSON 数据
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    print(f"📖 Loading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    prims = data.get("primitives", {})
    points = prims.get("points", [])
    lines = prims.get("lines", [])
    circles = prims.get("circles", [])

    # 2. 准备画布
    canvas = None
    h, w = 800, 800  # 默认尺寸，如果没找到原图

    if img_path and os.path.exists(img_path):
        print(f"🖼️ Loading background image: {img_path}")
        orig_img = cv2.imread(img_path)
        if orig_img is not None:
            h, w = orig_img.shape[:2]
            # 创建一个变淡的背景
            overlay = np.ones_like(orig_img) * 255  # 纯白图层
            # 将原图和纯白图层混合，使原图变淡
            canvas = cv2.addWeighted(orig_img, 1.0 - style['background_fade'], overlay, style['background_fade'], 0)
        else:
            print("Warning: Could not read background image. Using blank canvas.")
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        print("ℹ️ No background image provided. Using black canvas.")
        # 尝试从 JSON 中猜测画布大小（如果之前保存了的话），否则用默认
        canvas = np.zeros((h, w, 3), dtype=np.uint8)  # 纯黑背景

    print("🎨 Drawing vectors...")

    # 3. 绘制圆形 (绿色)
    for c in circles:
        center = tuple(c['center'])
        radius = c['radius']
        cv2.circle(canvas, center, radius, style['circle_color'], style['circle_thickness'], cv2.LINE_AA)

    # 4. 绘制直线 (红色)
    for l in lines:
        start = tuple(l['start'])
        end = tuple(l['end'])
        cv2.line(canvas, start, end, style['line_color'], style['line_thickness'], cv2.LINE_AA)

    # 5. 绘制关键点 (蓝色) - 最后绘制，确保盖在线上
    for p in points:
        center = tuple(p)
        # 外圈
        cv2.circle(canvas, center, style['point_radius'] + 2, (255, 255, 255), -1, cv2.LINE_AA)
        # 内芯
        cv2.circle(canvas, center, style['point_radius'], style['point_color'], -1, cv2.LINE_AA)

    # 6. 保存结果
    output_path = config['output_viz_path']
    cv2.imwrite(output_path, canvas)
    print(f"✅ Visualization saved to: {output_path}")
    print(f"   Stats: {len(points)} points, {len(lines)} lines, {len(circles)} circles plotted.")


if __name__ == "__main__":
    visualize_vectors(CONFIG)