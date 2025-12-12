import cv2
import numpy as np
import os
import json
import random
import math


class GeometricDatasetGenerator:
    def __init__(self, output_dir="geometry_dataset", width=384, height=384):
        """
        初始化生成器
        output_dir: 数据集保存根目录
        width/height: 图片尺寸 (Swin Transformer 建议 384x384)
        """
        self.output_dir = output_dir
        self.img_dir = os.path.join(output_dir, "images")
        self.mask_dir = os.path.join(output_dir, "masks")
        self.label_dir = os.path.join(output_dir, "labels")
        self.viz_dir = os.path.join(output_dir, "visualization")  # 方便肉眼检查

        # 创建所有目录
        for d in [self.img_dir, self.mask_dir, self.label_dir, self.viz_dir]:
            os.makedirs(d, exist_ok=True)

        self.width = width
        self.height = height

    # --------------------------
    # 1. 视觉渲染层 (Visual Layer)
    # --------------------------
    def _get_random_color(self):
        """随机墨水颜色"""
        colors = [
            (20, 20, 20), (30, 30, 30), (100, 50, 0), (60, 40, 20)
        ]
        return random.choice(colors)

    def _create_background(self):
        """生成纸张背景"""
        bg_type = random.choice(['white', 'yellow', 'grid', 'lined'])

        # 底色
        if bg_type == 'yellow':
            img = np.full((self.height, self.width, 3), (200, 235, 245), dtype=np.uint8)
        else:
            img = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        # 噪声
        noise = np.random.normal(0, 5, (self.height, self.width, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 格子/横线
        line_color = (200, 200, 200) if bg_type == 'white' else (180, 210, 220)
        if bg_type in ['grid', 'lined']:
            step = random.randint(30, 50)
            for y in range(0, self.height, step):
                cv2.line(img, (0, y), (self.width, y), line_color, 1)
            if bg_type == 'grid':
                for x in range(0, self.width, step):
                    cv2.line(img, (x, 0), (x, self.height), line_color, 1)

        # 污渍
        for _ in range(random.randint(20, 100)):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            cv2.circle(img, (x, y), 1, (150, 150, 150), -1)

        return img

    def _draw_hand_line(self, img, p1, p2, color):
        """模拟手抖直线"""
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-5: return img

        num_segments = int(dist / 5) + 1
        points = []

        # 波动参数
        fluctuation_idx = int(random.gauss(0.5, 0.15) * num_segments)
        fluctuation_amp = random.uniform(0, 8) * random.choice([-1, 1])

        for i in range(num_segments):
            t = i / (num_segments - 1) if num_segments > 1 else 0
            x = (1 - t) * x1 + t * x2
            y = (1 - t) * y1 + t * y2

            jitter_x = random.uniform(-1.0, 1.0)
            jitter_y = random.uniform(-1.0, 1.0)

            # 加入主要波动
            if abs(i - fluctuation_idx) < (num_segments / 4):
                decay = np.exp(-0.1 * (i - fluctuation_idx) ** 2)
                norm_x, norm_y = -(y2 - y1) / dist, (x2 - x1) / dist
                jitter_x += norm_x * fluctuation_amp * decay
                jitter_y += norm_y * fluctuation_amp * decay

            points.append((int(x + jitter_x), int(y + jitter_y)))

        thickness = random.randint(1, 2)
        for i in range(len(points) - 1):
            local_color = tuple([max(0, c - random.randint(0, 20)) for c in color])
            cv2.line(img, points[i], points[i + 1], local_color, thickness, lineType=cv2.LINE_AA)
        return img

    def _draw_hand_circle(self, img, center, radius, color):
        """模拟手绘圆"""
        cx, cy = center
        strokes = random.randint(1, 2)
        for _ in range(strokes):
            a = radius + random.uniform(-2, 2)
            b = radius + random.uniform(-2, 2)
            angle_offset = random.uniform(0, 6.28)
            start_angle = -random.uniform(0, 0.5)
            end_angle = 6.28 + random.uniform(0, 0.5)

            step = 0.05
            theta = start_angle
            prev_pt = None

            while theta < end_angle:
                raw_x = a * math.cos(theta)
                raw_y = b * math.sin(theta)
                rot_x = raw_x * math.cos(angle_offset) - raw_y * math.sin(angle_offset)
                rot_y = raw_x * math.sin(angle_offset) + raw_y * math.cos(angle_offset)
                curr_pt = (int(cx + rot_x + random.uniform(-1, 1)), int(cy + rot_y + random.uniform(-1, 1)))

                if prev_pt:
                    local_color = tuple([max(0, c - random.randint(0, 30)) for c in color])
                    cv2.line(img, prev_pt, curr_pt, local_color, 1, lineType=cv2.LINE_AA)
                prev_pt = curr_pt
                theta += step
        return img

    # --------------------------
    # 2. 掩码生成层 (Mask Layer)
    # --------------------------
    def _draw_gaussian(self, heatmap, center, sigma=5):
        """在热力图上画高斯点"""
        x, y = center
        tmp_size = sigma * 3
        ul = [int(x - tmp_size), int(y - tmp_size)]
        br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]

        if ul[0] >= self.width or ul[1] >= self.height or br[0] < 0 or br[1] < 0:
            return heatmap

        size = 2 * tmp_size + 1
        x0 = np.arange(0, size, 1, float)
        y0 = x0[:, np.newaxis]
        x00 = size // 2
        y00 = size // 2
        g = np.exp(- ((x0 - x00) ** 2 + (y0 - y00) ** 2) / (2 * sigma ** 2))

        g_x = max(0, -ul[0]), min(br[0], self.width) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], self.height) - ul[1]
        img_x = max(0, ul[0]), min(br[0], self.width)
        img_y = max(0, ul[1]), min(br[1], self.height)

        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
        return heatmap

    # --------------------------
    # 3. 核心生成逻辑 (Pipeline)
    # --------------------------
    def generate_batch(self, count=10):
        print(f"Starting generation of {count} samples...")

        for i in range(count):
            file_id = f"geo_{i:06d}"

            # --- 初始化画布 ---
            image = self._create_background()
            ink_color = self._get_random_color()

            # Mask: B=Points, G=Circles, R=Lines (OpenCV格式)
            mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            heatmap = np.zeros((self.height, self.width), dtype=np.float32)

            # JSON Annotation Structure
            annotation = {
                "lines": [],
                "circles": [],
                "intersections": []
            }

            # --- 几何数据生成 ---
            points = []
            margin = 50
            num_points = random.randint(3, 6)
            for _ in range(num_points):
                points.append((
                    random.randint(margin, self.width - margin),
                    random.randint(margin, self.height - margin)
                ))

            # 将所有生成的关键点记录为交点 (Intersections)
            # 在Mask中绘制高斯热力点 (B通道)
            for p in points:
                annotation["intersections"].append(p)
                self._draw_gaussian(heatmap, p, sigma=5)

            # 1. 生成线 (Lines)
            for j in range(len(points)):
                p1 = points[j]
                p2 = points[(j + 1) % len(points)]  # 闭环

                if random.random() > 0.15:  # 85%概率画这条边
                    # -> 绘制到 Visual Image (带手抖)
                    self._draw_hand_line(image, p1, p2, ink_color)

                    # -> 绘制到 Mask (R通道，实心，线宽3)
                    # 这里的颜色是 (B, G, R) -> (0, 0, 255)
                    cv2.line(mask, p1, p2, (0, 0, 255), 3)

                    # -> 记录 JSON
                    # 采样5个点
                    sample_pts = []
                    for t in np.linspace(0, 1, 5):
                        sx = int((1 - t) * p1[0] + t * p2[0])
                        sy = int((1 - t) * p1[1] + t * p2[1])
                        sample_pts.append([sx, sy])

                    annotation["lines"].append({
                        "endpoints": [p1, p2],
                        "sample_points": sample_pts
                    })

            # 2. 随机对角线
            if len(points) >= 4:
                p_start, p_end = points[0], points[2]
                self._draw_hand_line(image, p_start, p_end, ink_color)
                cv2.line(mask, p_start, p_end, (0, 0, 255), 3)

                # 采样点
                sample_pts = []
                for t in np.linspace(0, 1, 5):
                    sx = int((1 - t) * p_start[0] + t * p_end[0])
                    sy = int((1 - t) * p_start[1] + t * p_end[1])
                    sample_pts.append([sx, sy])

                annotation["lines"].append({
                    "endpoints": [p_start, p_end],
                    "sample_points": sample_pts
                })

            # 3. 生成圆 (Circles)
            if random.random() > 0.3:
                c_idx = random.randint(0, len(points) - 1)
                p_idx = (c_idx + 1) % len(points)
                center = points[c_idx]
                radius = int(math.hypot(center[0] - points[p_idx][0], center[1] - points[p_idx][1]))

                if 20 < radius < min(self.width, self.height) // 2.5:
                    # -> 绘制到 Visual Image
                    self._draw_hand_circle(image, center, radius, ink_color)

                    # -> 绘制到 Mask (G通道，实心，线宽3)
                    # 这里的颜色是 (B, G, R) -> (0, 255, 0)
                    cv2.circle(mask, center, radius, (0, 255, 0), 3)

                    # -> 记录 JSON
                    annotation["circles"].append({
                        "center": center,
                        "radius": radius
                    })

            # --- 合并Mask通道 ---
            # 将 Heatmap 归一化到 0-255 并放入 Blue 通道
            heatmap_uint8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
            mask[:, :, 0] = heatmap_uint8  # B通道

            # --- 保存文件 ---
            cv2.imwrite(os.path.join(self.img_dir, f"{file_id}.jpg"), image)
            cv2.imwrite(os.path.join(self.mask_dir, f"{file_id}.png"), mask)
            with open(os.path.join(self.label_dir, f"{file_id}.json"), 'w') as f:
                json.dump(annotation, f, indent=2)

            # --- 生成可视化预览图 (可选) ---
            # 将 mask 叠加在 原图 上，方便你肉眼看对不对
            # Mask转彩色叠加：Mask是有颜色的，直接 addWeighted
            viz = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
            cv2.imwrite(os.path.join(self.viz_dir, f"{file_id}_viz.jpg"), viz)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{count} images")


if __name__ == "__main__":
    # 生成 100 张作为示例
    # 尺寸设为 384x384 (适合 Swin Transformer Patch Partition)
    generator = GeometricDatasetGenerator(output_dir="dataset_v1", width=384, height=384)
    generator.generate_batch(count=100)
    print("Done! Check 'dataset_v1' folder.")