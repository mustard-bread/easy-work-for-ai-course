import cv2
import numpy as np
import json
import math
from skimage.morphology import skeletonize

# ================= 配置区域 =================
CONFIG = {
    # 这里输入你已经画好的结果图路径（例如 result_final.jpg）
    "input_image_path": r"C:\Users\Administrator\PycharmProjects\geo_detect\result_final.jpg",
    "output_json": "final_vectors.json",

    # 颜色阈值 (HSV空间)，如果识别不准可以微调
    # OpenCV HSV范围: H(0-180), S(0-255), V(0-255)
    "colors": {
        "blue_point": {"lower": np.array([100, 100, 100]), "upper": np.array([140, 255, 255])},  # 蓝色点
        "green_circle": {"lower": np.array([40, 100, 100]), "upper": np.array([90, 255, 255])},  # 绿色圆
        # 红色通常分布在0-10和170-180两个区间
        "red_line_1": {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},
        "red_line_2": {"lower": np.array([170, 100, 100]), "upper": np.array([180, 255, 255])}
    },

    # 参数微调
    "snap_distance": 20,  # 线段端点吸附到点的最大距离
    "min_line_length": 15,  # 忽略过短的噪点线
    "max_line_gap": 20  # 允许线段断裂的最大距离
}


# ===========================================

class ColorToVector:
    def __init__(self, config):
        self.cfg = config

    def get_color_masks(self, img_hsv):
        """根据HSV范围提取红、绿、蓝三个Mask"""
        # 1. 蓝色 (点)
        mask_blue = cv2.inRange(img_hsv, self.cfg['colors']['blue_point']['lower'], self.cfg['colors']['blue_point']['upper'])

        # 2. 绿色 (圆)
        mask_green = cv2.inRange(img_hsv, self.cfg['colors']['green_circle']['lower'], self.cfg['colors']['green_circle']['upper'])

        # 3. 红色 (线) - 需要合并两个区间
        mask_red1 = cv2.inRange(img_hsv, self.cfg['colors']['red_line_1']['lower'], self.cfg['colors']['red_line_1']['upper'])
        mask_red2 = cv2.inRange(img_hsv, self.cfg['colors']['red_line_2']['lower'], self.cfg['colors']['red_line_2']['upper'])
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 简单的形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)  # 闭运算连接断裂的线

        return mask_blue, mask_green, mask_red

    def extract_points(self, mask):
        """从蓝色Mask提取点坐标"""
        points = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append([cx, cy])
        return points

    def extract_circles(self, mask):
        """从绿色Mask提取圆信息"""
        circles = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue  # 忽略噪点
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circles.append({
                "center": [int(x), int(y)],
                "radius": int(radius)
            })
        return circles

    def extract_lines(self, mask, points):
        """从红色Mask提取直线，并吸附到点"""
        # 1. 骨架化：把粗红线变成单像素宽的线
        # scikit-image 的 skeletonize 需要 0/1 输入
        binary_mask = mask > 0
        skeleton = skeletonize(binary_mask).astype(np.uint8) * 255

        # 2. 霍夫直线变换检测线段
        lines_p = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=10,
            minLineLength=self.cfg['min_line_length'],
            maxLineGap=self.cfg['max_line_gap']
        )

        vector_lines = []
        if lines_p is not None:
            for line in lines_p:
                x1, y1, x2, y2 = line[0]
                p1 = [int(x1), int(y1)]
                p2 = [int(x2), int(y2)]

                # 3. 吸附逻辑：让线头自动连到最近的蓝点上
                p1 = self.snap_to_nearest(p1, points)
                p2 = self.snap_to_nearest(p2, points)

                vector_lines.append({"start": p1, "end": p2})

        return vector_lines

    def snap_to_nearest(self, point, anchors):
        """如果距离足够近，将点吸附到锚点(蓝点)"""
        if not anchors: return point
        p_arr = np.array(point)
        anchors_arr = np.array(anchors)
        dists = np.linalg.norm(anchors_arr - p_arr, axis=1)
        min_idx = np.argmin(dists)

        if dists[min_idx] < self.cfg['snap_distance']:
            return list(map(int, anchors[min_idx]))
        return point

    def run(self):
        print(f"🖼️ Reading image: {self.cfg['input_image_path']}")
        img = cv2.imread(self.cfg['input_image_path'])
        if img is None:
            print("Error: Image not found!")
            return

        # 转为HSV空间以便分割颜色
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1. 提取Mask
        mask_blue, mask_green, mask_red = self.get_color_masks(img_hsv)

        # 2. 提取图元
        print("🔍 Extracting primitives...")
        points = self.extract_points(mask_blue)
        circles = self.extract_circles(mask_green)
        lines = self.extract_lines(mask_red, points)  # 把lines吸附到points上

        print(f"📊 Detect: {len(points)} Points, {len(circles)} Circles, {len(lines)} Lines")

        # 3. 保存JSON
        result_json = {
            "source_image": self.cfg['input_image_path'],
            "primitives": {
                "points": points,
                "circles": circles,
                "lines": lines
            }
        }

        with open(self.cfg['output_json'], 'w') as f:
            json.dump(result_json, f, indent=4)

        print(f"✅ JSON saved to {self.cfg['output_json']}")

        # (可选) 验证可视化：画出来看看吸附得对不对
        debug_img = np.zeros_like(img)
        for c in circles:
            cv2.circle(debug_img, tuple(c['center']), c['radius'], (0, 255, 0), 2)
        for l in lines:
            cv2.line(debug_img, tuple(l['start']), tuple(l['end']), (0, 0, 255), 2)
        for p in points:
            cv2.circle(debug_img, tuple(p), 5, (255, 0, 0), -1)
        cv2.imwrite("debug_verify.jpg", debug_img)
        print("📸 Saved debug_verify.jpg for checking.")


if __name__ == "__main__":
    app = ColorToVector(CONFIG)
    app.run()