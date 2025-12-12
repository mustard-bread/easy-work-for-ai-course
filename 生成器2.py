import cv2
import numpy as np
import os
import json
import random
import math


class GeometricDatasetGenerator:
    def __init__(self, output_dir="geometry_dataset_refined", width=384, height=384):
        self.output_dir = output_dir
        self.img_dir = os.path.join(output_dir, "images")
        self.mask_dir = os.path.join(output_dir, "masks")
        self.label_dir = os.path.join(output_dir, "labels")
        self.viz_dir = os.path.join(output_dir, "visualization")

        for d in [self.img_dir, self.mask_dir, self.label_dir, self.viz_dir]:
            os.makedirs(d, exist_ok=True)

        self.width = width
        self.height = height
        # 定义像素与厘米的关系 (估算 384px = 10cm)
        self.px_per_cm = 40

        # --------------------------

    # 1. 视觉渲染层 (Visual Layer)
    # --------------------------
    def _get_random_color(self):
        colors = [(20, 20, 20), (30, 30, 30), (100, 50, 0), (60, 40, 20)]
        return random.choice(colors)

    def _create_background(self):
        bg_type = random.choice(['white', 'yellow', 'grid', 'lined'])
        if bg_type == 'yellow':
            img = np.full((self.height, self.width, 3), (200, 235, 245), dtype=np.uint8)
        else:
            img = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        noise = np.random.normal(0, 5, (self.height, self.width, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        line_color = (200, 200, 200) if bg_type == 'white' else (180, 210, 220)
        if bg_type in ['grid', 'lined']:
            step = random.randint(30, 50)
            for y in range(0, self.height, step):
                cv2.line(img, (0, y), (self.width, y), line_color, 1)
            if bg_type == 'grid':
                for x in range(0, self.width, step):
                    cv2.line(img, (x, 0), (x, self.height), line_color, 1)

        for _ in range(random.randint(20, 100)):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            cv2.circle(img, (x, y), 1, (150, 150, 150), -1)
        return img

    def _draw_hand_line(self, img, p1, p2, color):
        """
        模拟手抖直线 (更新版：符合正态分布误差)
        误差范围：0cm (标准) 到 ~0.3cm (极大值，3-sigma)
        """
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-5: return img

        # 切分得更细一点，以便表现平滑的波动
        num_segments = int(dist / 3) + 2
        points = []

        # --- 新的波动幅度计算逻辑 ---
        target_max_error_cm = 0.3
        # 3-sigma 边界对应的像素值 (约 12px)
        three_sigma_limit_px = target_max_error_cm * self.px_per_cm
        # 标准差
        sigma_px = three_sigma_limit_px / 3.0

        # 使用半正态分布 (取绝对值)，使得大部分线条误差很小
        fluctuation_amp_abs = abs(random.gauss(0, sigma_px))
        # 随机波动方向
        fluctuation_amp = fluctuation_amp_abs * random.choice([-1, 1])

        # 波动中心位置 (依然是中间偏随机)
        fluctuation_idx = int(random.gauss(0.5, 0.1) * num_segments)

        # 计算法向量，用于施加波动
        dx, dy = x2 - x1, y2 - y1
        norm_x, norm_y = -dy / dist, dx / dist

        for i in range(num_segments):
            t = i / (num_segments - 1) if num_segments > 1 else 0
            # 理想直线上的点
            x_ideal = (1 - t) * x1 + t * x2
            y_ideal = (1 - t) * y1 + t * y2

            # 1. 基础的高频小抖动 (墨水扩散感，很小)
            jitter_x = random.uniform(-0.5, 0.5)
            jitter_y = random.uniform(-0.5, 0.5)

            # 2. 施加主要的钟形波动
            # 使用高斯函数作为衰减系数，只在 fluctuation_idx 附近产生偏移
            # 控制波峰宽度
            width_factor = num_segments / 5.0
            decay = np.exp(-0.5 * ((i - fluctuation_idx) / width_factor) ** 2)

            # 沿着法向量方向偏移
            offset_x = norm_x * fluctuation_amp * decay
            offset_y = norm_y * fluctuation_amp * decay

            final_x = int(x_ideal + offset_x + jitter_x)
            final_y = int(y_ideal + offset_y + jitter_y)
            points.append((final_x, final_y))

        # 绘制
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
            # 稍微增加一点椭圆离心率的随机性
            a = radius + random.uniform(-3, 3)
            b = radius + random.uniform(-3, 3)
            angle_offset = random.uniform(0, 6.28)
            start_angle = -random.uniform(0, 0.3)
            end_angle = 6.28 + random.uniform(0, 0.3)

            step = 0.04  # 步长更小一点更平滑
            theta = start_angle
            prev_pt = None

            while theta < end_angle:
                raw_x = a * math.cos(theta)
                raw_y = b * math.sin(theta)
                rot_x = raw_x * math.cos(angle_offset) - raw_y * math.sin(angle_offset)
                rot_y = raw_x * math.sin(angle_offset) + raw_y * math.cos(angle_offset)

                # 圆本身的抖动也要符合整体风格，稍微小一点
                curr_pt = (int(cx + rot_x + random.uniform(-0.5, 0.5)),
                           int(cy + rot_y + random.uniform(-0.5, 0.5)))

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
        x, y = center
        tmp_size = sigma * 3
        ul = [int(x - tmp_size), int(y - tmp_size)]
        br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
        if ul[0] >= self.width or ul[1] >= self.height or br[0] < 0 or br[1] < 0: return heatmap
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
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        return heatmap

    # --------------------------
    # 3. 核心生成逻辑 (Pipeline)
    # --------------------------
    def generate_batch(self, count=10):
        print(f"Starting generation of {count} samples with refined line style...")
        for i in range(count):
            file_id = f"geo_refined_{i:06d}"
            image = self._create_background()
            ink_color = self._get_random_color()
            mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            heatmap = np.zeros((self.height, self.width), dtype=np.float32)
            annotation = {"lines": [], "circles": [], "intersections": []}

            points = []
            margin = 60
            num_points = random.randint(3, 6)
            for _ in range(num_points):
                points.append((random.randint(margin, self.width - margin),
                               random.randint(margin, self.height - margin)))

            # Draw Points (Heatmap Blue Channel)
            for p in points:
                annotation["intersections"].append(p)
                self._draw_gaussian(heatmap, p, sigma=5)

            # Draw Lines (Red Channel)
            for j in range(len(points)):
                p1 = points[j]
                p2 = points[(j + 1) % len(points)]
                if random.random() > 0.2:
                    self._draw_hand_line(image, p1, p2, ink_color)
                    cv2.line(mask, p1, p2, (0, 0, 255), 3)  # BGR -> R
                    sample_pts = []
                    for t in np.linspace(0, 1, 5):
                        sample_pts.append([int((1 - t) * p1[0] + t * p2[0]), int((1 - t) * p1[1] + t * p2[1])])
                    annotation["lines"].append({"endpoints": [p1, p2], "sample_points": sample_pts})

            # Random Diagonal Line
            if len(points) >= 4 and random.random() > 0.3:
                p_start, p_end = points[0], points[2]
                self._draw_hand_line(image, p_start, p_end, ink_color)
                cv2.line(mask, p_start, p_end, (0, 0, 255), 3)
                sample_pts = []
                for t in np.linspace(0, 1, 5):
                    sample_pts.append([int((1 - t) * p_start[0] + t * p_end[0]), int((1 - t) * p_start[1] + t * p_end[1])])
                annotation["lines"].append({"endpoints": [p_start, p_end], "sample_points": sample_pts})

            # Draw Circles (Green Channel)
            if random.random() > 0.4:
                c_idx = random.randint(0, len(points) - 1)
                p_idx = (c_idx + 1) % len(points)
                center = points[c_idx]
                radius = int(math.hypot(center[0] - points[p_idx][0], center[1] - points[p_idx][1]))
                if 30 < radius < min(self.width, self.height) // 3:
                    self._draw_hand_circle(image, center, radius, ink_color)
                    cv2.circle(mask, center, radius, (0, 255, 0), 3)  # BGR -> G
                    annotation["circles"].append({"center": center, "radius": radius})

            heatmap_uint8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
            mask[:, :, 0] = heatmap_uint8  # B Channel

            cv2.imwrite(os.path.join(self.img_dir, f"{file_id}.jpg"), image)
            cv2.imwrite(os.path.join(self.mask_dir, f"{file_id}.png"), mask)
            with open(os.path.join(self.label_dir, f"{file_id}.json"), 'w') as f:
                json.dump(annotation, f, indent=2)
            viz = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
            cv2.imwrite(os.path.join(self.viz_dir, f"{file_id}_viz.jpg"), viz)

            if (i + 1) % 10 == 0: print(f"Generated {i + 1}/{count} images")


if __name__ == "__main__":
    # 生成 50 张查看效果
    generator = GeometricDatasetGenerator(output_dir="dataset_refined_v2", width=384, height=384)
    generator.generate_batch(count=50)
    print("Done! Check 'dataset_refined_v2/visualization' to see the new line styles.")