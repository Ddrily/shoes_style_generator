import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import matplotlib.pyplot as plt
import json
from training3 import UnifiedShoeDesigner
from choice import filter_clear_images, interactive_filter_adjustment  # 导入筛选功能


def generate_preset_shoes():
    """批量生成预设组合的鞋子"""

    # 检查模型是否存在
    if not os.path.exists("model_complete.pth"):
        print("错误: 未找到训练好的模型 'model_complete.pth'")
        print("请先运行 training3.py 完成模型训练")
        return

    if not os.path.exists("vocab.json"):
        print("错误: 未找到词汇表文件 'vocab.json'")
        return

    try:
        # 加载设计器
        print("加载鞋子设计器...")
        designer = UnifiedShoeDesigner()
        print("设计器加载成功！")

        color = "red"
        style = "shoe"

        # 预设的颜色和款式组合
        preset_combinations = [
            # (颜色, 款式)
            (color, style),
        ]

        # 创建输出目录
        output_dir = "batch_generated_shoes"
        os.makedirs(output_dir, exist_ok=True)

        print(f"开始批量生成 {len(preset_combinations)} 种鞋子组合...")

        # 批量生成
        successful_generations = 0

        for i, (color, style) in enumerate(preset_combinations):
            try:
                print(f"生成中... ({i + 1}/{len(preset_combinations)}): {color} {style}")

                # 为每种组合生成16张不同的图片
                for j in range(16):
                    # 生成鞋子图像 - 每次生成1张，通过随机种子实现多样性
                    shoe_image = designer.design_shoe(color, style, num_variations=1)

                    # 保存单个图像
                    plt.figure(figsize=(6, 6))
                    plt.imshow(shoe_image)
                    plt.axis('off')
                    plt.title(f"{color} {style} - {j + 1}", fontsize=14, pad=10)

                    filename = f"{output_dir}/{color}_{style}_{j + 1}.png"
                    plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
                    plt.close()

                    print(f"  ✓ 已保存: {filename}")

                successful_generations += 1

            except Exception as e:
                print(f"  ✗ 生成失败 {color} {style}: {e}")

        # 创建组合预览图
        create_preview_grid(output_dir, preset_combinations)

        print(f"\n批量生成完成！")
        print(f"成功生成: {successful_generations}/{len(preset_combinations)}")
        print(f"总共生成图片: {successful_generations * 16} 张")
        print(f"图像保存在: {output_dir}/")

        # 生成完成后自动调用筛选功能
        print("\n" + "=" * 50)
        print("开始筛选清晰图片...")
        auto_filter_generated_images()

    except Exception as e:
        print(f"生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def auto_filter_generated_images():
    """自动筛选生成的图片"""
    try:
        # 检查是否有生成的图片
        if not os.path.exists("batch_generated_shoes"):
            print("错误: 未找到生成的图片目录 'batch_generated_shoes'")
            print("请先运行生成功能")
            return

        # 获取筛选参数
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
            print("\n⚠ 未找到符合标准的清晰图片")

    except Exception as e:
        print(f"筛选过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def create_preview_grid(output_dir, combinations):
    """创建所有生成图像的预览网格"""
    try:
        # 计算网格大小
        n_cols = 4  # 每行3个
        n_rows = (len(combinations) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for idx, (color, style) in enumerate(combinations):
            if idx < len(axes):
                img_path = f"{output_dir}/{color}_{style}.png"

                if os.path.exists(img_path):
                    img = plt.imread(img_path)
                    axes[idx].imshow(img)
                    axes[idx].set_title(f"{color} {style}", fontsize=10)
                else:
                    axes[idx].text(0.5, 0.5, f"Missing\n{color} {style}",
                                   ha='center', va='center', fontsize=8)

                axes[idx].axis('off')

        # 隐藏多余的子图
        for idx in range(len(combinations), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_shoes_preview.png", dpi=200, bbox_inches='tight')
        plt.close()

        print(f"预览图已保存: {output_dir}/all_shoes_preview.png")

    except Exception as e:
        print(f"创建预览图时出错: {e}")


if __name__ == "__main__":
    print("=== 鞋子批量生成器 ===")
    print("批量生成预设组合并自动筛选清晰图片")

    generate_preset_shoes()