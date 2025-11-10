import os

# 设置环境变量解决内存泄漏问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'  # 解决KMeans内存泄漏问题

import json
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端避免显示问题
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def remove_background_and_detect_color(image_path):
    """改进的背景移除和颜色检测"""
    try:
        image = Image.open(image_path)
        original_size = image.size
        image = image.resize((150, 150))  # 适当增大尺寸以保留更多细节
        image_array = np.array(image)

        # 转换为多个颜色空间
        hsv_image = image.convert('HSV')
        hsv_array = np.array(hsv_image)

        # 多种背景检测方法
        # 1. HSV空间的白色/浅色检测
        white_mask_hsv = (hsv_array[:, :, 1] < 40) & (hsv_array[:, :, 2] > 200)

        # 2. RGB空间的白色检测
        rgb_white_mask = (image_array[:, :, 0] > 220) & (image_array[:, :, 1] > 220) & (image_array[:, :, 2] > 220)

        # 3. 基于边缘的检测 - 找到可能的主体区域
        from scipy import ndimage
        gray = np.mean(image_array, axis=2)
        edges = ndimage.sobel(gray)
        edge_mask = edges > np.percentile(edges, 75)

        # 结合多种方法创建前景掩码
        background_mask = white_mask_hsv | rgb_white_mask
        # 边缘区域很可能是前景
        foreground_mask = ~background_mask | edge_mask

        # 使用连通组件分析去除小噪点
        from scipy import ndimage as ndi
        labeled_mask, num_features = ndi.label(foreground_mask)
        component_sizes = np.bincount(labeled_mask.ravel())

        # 如果最大的连通组件太小，使用中心区域
        if len(component_sizes) > 1 and component_sizes[1:].max() < 500:
            h, w = image_array.shape[:2]
            center_mask = np.zeros((h, w), dtype=bool)
            center_size_h, center_size_w = h // 2, w // 2
            center_start_h, center_start_w = h // 4, w // 4
            center_mask[center_start_h:center_start_h + center_size_h,
            center_start_w:center_start_w + center_size_w] = True
            foreground_mask = center_mask
        else:
            # 保留足够大的连通组件
            min_size = 100
            for i in range(1, num_features + 1):
                if component_sizes[i] < min_size:
                    foreground_mask[labeled_mask == i] = False

        # 提取前景像素
        foreground_pixels = image_array[foreground_mask]

        if len(foreground_pixels) < 10:
            # 如果前景像素太少，使用整个图像的中心区域
            h, w = image_array.shape[:2]
            center_h, center_w = h // 2, w // 2
            center_size = min(h, w) // 2
            center_mask = np.zeros((h, w), dtype=bool)
            center_mask[center_h - center_size:center_h + center_size,
            center_w - center_size:center_w + center_size] = True
            foreground_pixels = image_array[center_mask]

        if len(foreground_pixels) == 0:
            return (128, 128, 128), None  # 返回灰色而不是黑色

        # 使用K-means聚类找到主要颜色
        if len(foreground_pixels) > 1000:
            # 抽样以加快处理速度
            indices = np.random.choice(len(foreground_pixels), 1000, replace=False)
            sample_pixels = foreground_pixels[indices]
        else:
            sample_pixels = foreground_pixels

        # 尝试K-means聚类
        try:
            # 使用更少的聚类中心和更少的迭代次数来减少内存使用
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=5, max_iter=50)
            kmeans.fit(sample_pixels)

            # 获取主要簇的中心
            cluster_centers = kmeans.cluster_centers_.astype(int)
            # 选择最大的簇作为主要颜色
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            main_cluster_idx = labels[np.argmax(counts)]
            dominant_color = tuple(cluster_centers[main_cluster_idx])
        except:
            # 如果K-means失败，使用中位数
            dominant_color = tuple(np.median(foreground_pixels, axis=0).astype(int))

        return dominant_color, foreground_pixels

    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return (128, 128, 128), None


def improved_rgb_to_color_name(rgb):
    """改进的颜色名称识别"""
    r, g, b = rgb

    # 计算亮度和饱和度
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    brightness = (max_val + min_val) / 2
    saturation = max_val - min_val if max_val != 0 else 0

    # 黑色、白色、灰色检测
    if saturation < 30:  # 低饱和度
        if brightness > 200:
            return 'white'
        elif brightness > 100:
            return 'light_gray'
        elif brightness > 50:
            return 'gray'
        else:
            return 'black'

    # 主要颜色检测（使用更精确的阈值）
    if r > max(g, b) + 30:  # 红色系
        if r > 180 and g < 100 and b < 100:
            return 'red'
        elif r > 120:
            if g < 80 and b < 80:
                return 'dark_red'
            elif g > 100 and b < 100:
                return 'orange_red'
        return 'reddish'

    elif g > max(r, b) + 30:  # 绿色系
        if g > 180 and r < 100 and b < 100:
            return 'green'
        elif g > 120:
            if r < 80 and b < 80:
                return 'dark_green'
            elif r > 100 and b < 100:
                return 'yellow_green'
        return 'greenish'

    elif b > max(r, g) + 30:  # 蓝色系
        if b > 180 and r < 100 and g < 100:
            return 'blue'
        elif b > 120:
            if r < 80 and g < 80:
                return 'dark_blue'
            elif r > 100 and g < 150:
                return 'purple_blue'
        return 'bluish'

    # 混合颜色检测
    elif r > 150 and g > 150 and b < 100:  # 黄色
        if r > 200 and g > 200:
            return 'bright_yellow'
        else:
            return 'yellow'

    elif r > 150 and g > 100 and b > 150:  # 粉色/紫色
        if r > g and r > b:
            return 'pink'
        else:
            return 'purple'

    elif r > 150 and g < 100 and b > 150:  # 紫色
        return 'purple'

    elif 100 <= r <= 180 and 50 <= g <= 120 and b < 80:  # 棕色
        return 'brown'

    elif r > 180 and g > 180 and b > 180:  # 白色
        return 'white'

    elif r < 50 and g < 50 and b < 50:  # 黑色
        return 'black'

    elif r > 180 and g > 120 and b < 100:  # 橙色
        return 'orange'

    else:
        # 基于主要颜色成分的通用分类
        if r > g and r > b:
            return 'reddish'
        elif g > r and g > b:
            return 'greenish'
        elif b > r and b > g:
            return 'bluish'
        else:
            return 'multicolor'


def create_accurate_metadata_improved(root_folder, sample_check=False):
    """创建准确的元数据（改进版本）"""
    metadata = {}
    color_stats = Counter()

    print("开始分析图像实际颜色（改进版本）...")

    # 如果需要抽样检查，只处理部分文件
    sample_files = None
    if sample_check:
        sample_files = []
        for style in ['Boot', 'Sandal', 'Shoe']:
            style_folder = os.path.join(root_folder, style)
            if os.path.exists(style_folder):
                files = [f for f in os.listdir(style_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                sample_files.extend([(style, f) for f in files[:30]])

    for style in ['Boot', 'Sandal', 'Shoe']:
        style_folder = os.path.join(root_folder, style)

        if os.path.exists(style_folder):
            print(f"分析 {style} 文件夹...")

            if sample_check:
                files = [f for style_dir, f in sample_files if style_dir == style]
            else:
                files = [f for f in os.listdir(style_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for i, filename in enumerate(files):
                if i % 100 == 0:
                    print(f"  已处理 {i}/{len(files)} 张图片")

                image_path = os.path.join(style_folder, filename)

                # 提取实际颜色（去除背景）
                dominant_rgb, foreground_pixels = remove_background_and_detect_color(image_path)
                actual_color = improved_rgb_to_color_name(dominant_rgb)

                # 转换为Python原生类型以便JSON序列化
                dominant_rgb_list = [int(x) for x in dominant_rgb]

                metadata[filename] = {
                    'color': actual_color,
                    'style': style,
                    'detected_rgb': dominant_rgb_list
                }

                color_stats[actual_color] += 1

    # 保存元数据
    output_file = 'shoe_metadata_improved_colors.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n颜色统计:")
    total_images = sum(color_stats.values())
    for color, count in color_stats.most_common():
        percentage = (count / total_images) * 100
        print(f"  {color}: {count} 张 ({percentage:.1f}%)")

    print(f"\n改进版元数据已保存到: {output_file}")
    return metadata


def visualize_color_distribution(metadata, output_path='color_distribution.png'):
    """可视化颜色分布"""
    color_counts = Counter([data['color'] for data in metadata.values()])

    plt.figure(figsize=(12, 8))
    colors = list(color_counts.keys())
    counts = list(color_counts.values())

    # 创建颜色条
    color_bars = []
    for color_name in colors:
        # 简化颜色映射
        color_map = {
            'red': 'red', 'dark_red': 'darkred', 'reddish': 'lightcoral',
            'green': 'green', 'dark_green': 'darkgreen', 'greenish': 'lightgreen',
            'blue': 'blue', 'dark_blue': 'darkblue', 'bluish': 'lightblue',
            'yellow': 'yellow', 'bright_yellow': 'gold',
            'orange': 'orange', 'orange_red': 'orangered',
            'purple': 'purple', 'pink': 'pink',
            'brown': 'brown', 'black': 'black', 'white': 'white',
            'gray': 'gray', 'light_gray': 'lightgray',
            'multicolor': 'violet'
        }
        color_bars.append(color_map.get(color_name, 'gray'))

    bars = plt.bar(colors, counts, color=color_bars)
    plt.xlabel('颜色类别')
    plt.ylabel('数量')
    plt.title('鞋子颜色分布')
    plt.xticks(rotation=45, ha='right')

    # 在柱子上添加数量标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"颜色分布图已保存到: {output_path}")


# 使用示例
if __name__ == "__main__":
    root_folder = "D:/python/deep learn/archive/Shoe vs Sandal vs Boot Dataset"

    print("=== 改进的颜色检测工作流程 ===")

    try:
        # 直接处理所有图片
        print("\n开始处理所有图片...")
        accurate_metadata = create_accurate_metadata_improved(root_folder, sample_check=False)
        print("\n完成！现在可以使用准确的元数据训练模型了。")

        # 创建可视化
        visualize_color_distribution(accurate_metadata)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")