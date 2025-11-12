import os
import cv2
import numpy as np
import shutil
from pathlib import Path


def calculate_sharpness(image_path):
    """
    计算图像清晰度（使用拉普拉斯方差法）
    返回值越高表示图像越清晰
    """
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return 0

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算拉普拉斯方差
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        return laplacian_var
    except Exception as e:
        print(f"计算清晰度时出错 {image_path}: {e}")
        return 0


def analyze_brightness_contrast(image_path):
    """
    分析图像的亮度和对比度
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0, 0

        # 转换为HSV色彩空间分析亮度
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])

        # 计算对比度（标准差）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)

        return brightness, contrast
    except Exception as e:
        print(f"分析亮度对比度时出错 {image_path}: {e}")
        return 0, 0


def filter_clear_images(input_dir="batch_generated_shoes", output_dir="clear_shoes",
                        sharpness_threshold=100, brightness_range=(30, 220),
                        min_contrast=25):
    """
    筛选清晰的鞋子图片

    参数:
    - input_dir: 输入图片目录
    - output_dir: 清晰图片输出目录
    - sharpness_threshold: 清晰度阈值
    - brightness_range: 亮度范围 (min, max)
    - min_contrast: 最小对比度
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有PNG图片
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    if not image_files:
        print(f"在目录 {input_dir} 中未找到PNG图片")
        return

    print(f"找到 {len(image_files)} 张图片，开始筛选...")

    # 分析每张图片
    results = []
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)

        # 计算各项指标
        sharpness = calculate_sharpness(image_path)
        brightness, contrast = analyze_brightness_contrast(image_path)

        # 判断是否清晰
        is_clear = (sharpness >= sharpness_threshold and
                    brightness_range[0] <= brightness <= brightness_range[1] and
                    contrast >= min_contrast)

        results.append({
            'filename': image_file,
            'path': image_path,
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'is_clear': is_clear
        })

        status = "✓" if is_clear else "✗"
        print(f"{status} {image_file}: 清晰度={sharpness:.1f}, 亮度={brightness:.1f}, 对比度={contrast:.1f}")

    # 筛选清晰图片
    clear_images = [r for r in results if r['is_clear']]
    unclear_images = [r for r in results if not r['is_clear']]

    print(f"\n筛选结果:")
    print(f"清晰图片: {len(clear_images)} 张")
    print(f"模糊图片: {len(unclear_images)} 张")

    # 复制清晰图片到输出目录
    for result in clear_images:
        shutil.copy2(result['path'], os.path.join(output_dir, result['filename']))

    # 生成筛选报告
    generate_filter_report(results, output_dir)

    # 创建清晰图片预览
    create_clear_preview(clear_images, output_dir)

    return clear_images


def generate_filter_report(results, output_dir):
    """生成筛选报告"""
    report_path = os.path.join(output_dir, "filter_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("鞋子图片清晰度筛选报告\n")
        f.write("=" * 50 + "\n\n")

        # 清晰图片
        f.write("清晰图片:\n")
        f.write("-" * 30 + "\n")
        clear_images = [r for r in results if r['is_clear']]
        for result in clear_images:
            f.write(f"{result['filename']}: 清晰度={result['sharpness']:.1f}, "
                    f"亮度={result['brightness']:.1f}, 对比度={result['contrast']:.1f}\n")

        # 模糊图片
        f.write(f"\n模糊图片:\n")
        f.write("-" * 30 + "\n")
        unclear_images = [r for r in results if not r['is_clear']]
        for result in unclear_images:
            f.write(f"{result['filename']}: 清晰度={result['sharpness']:.1f}, "
                    f"亮度={result['brightness']:.1f}, 对比度={result['contrast']:.1f}\n")

        # 统计信息
        f.write(f"\n统计信息:\n")
        f.write("-" * 30 + "\n")
        f.write(f"总图片数: {len(results)}\n")
        f.write(f"清晰图片: {len(clear_images)}\n")
        f.write(f"模糊图片: {len(unclear_images)}\n")
        f.write(f"清晰率: {len(clear_images) / len(results) * 100:.1f}%\n")

        # 质量分布
        sharpness_values = [r['sharpness'] for r in results]
        f.write(f"\n清晰度分布:\n")
        f.write(f"最高: {max(sharpness_values):.1f}\n")
        f.write(f"最低: {min(sharpness_values):.1f}\n")
        f.write(f"平均: {np.mean(sharpness_values):.1f}\n")
        f.write(f"中位数: {np.median(sharpness_values):.1f}\n")

    print(f"筛选报告已保存: {report_path}")


def create_clear_preview(clear_images, output_dir):
    """创建清晰图片的预览网格"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        if not clear_images:
            print("没有清晰图片可创建预览")
            return

        # 计算网格布局
        n_images = len(clear_images)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(16, 4 * n_rows))

        for i, result in enumerate(clear_images):
            ax = plt.subplot(n_rows, n_cols, i + 1)

            # 读取并显示图像
            img = plt.imread(result['path'])
            ax.imshow(img)

            # 设置标题（包含质量信息）
            title = f"{result['filename']}\n清晰度: {result['sharpness']:.1f}"
            ax.set_title(title, fontsize=8)
            ax.axis('off')

        plt.tight_layout()
        preview_path = os.path.join(output_dir, "clear_shoes_preview.png")
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"清晰图片预览已保存: {preview_path}")

    except Exception as e:
        print(f"创建预览时出错: {e}")


def interactive_filter_adjustment():
    """交互式调整筛选参数"""
    print("=== 交互式清晰度筛选调整 ===")

    # 测试一张图片来获取参数范围
    test_dir = "batch_generated_shoes"
    if os.path.exists(test_dir):
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.png')]
        if image_files:
            test_image = os.path.join(test_dir, image_files[0])
            sharpness = calculate_sharpness(test_image)
            brightness, contrast = analyze_brightness_contrast(test_image)

            print(f"示例图片质量指标:")
            print(f"清晰度: {sharpness:.1f}")
            print(f"亮度: {brightness:.1f}")
            print(f"对比度: {contrast:.1f}")


    sharpness_threshold = int(280),#清晰度阈值
    min_brightness = int(30),#最小亮度
    max_brightness = int(220)#最大亮度
    min_contrast = int(60)#最小对比度

    return sharpness_threshold, (min_brightness, max_brightness), min_contrast


if __name__ == "__main__":
    print("=== 鞋子图片清晰度筛选器 ===")

    # 交互式调整参数
    sharpness_threshold, brightness_range, min_contrast = interactive_filter_adjustment()

    # 执行筛选
    clear_images = filter_clear_images(
        input_dir="batch_generated_shoes",
        output_dir="clear_shoes",
        sharpness_threshold=sharpness_threshold,
        brightness_range=brightness_range,
        min_contrast=min_contrast
    )

    if clear_images:
        print(f"\n✓ 筛选完成！共找到 {len(clear_images)} 张清晰图片")
        print(f"清晰图片保存在: clear_shoes/")
        print(f"查看筛选报告: clear_shoes/filter_report.txt")
    else:
        print("\n⚠ 未找到符合标准的清晰图片，建议调整筛选参数")