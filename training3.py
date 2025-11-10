import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from collections import defaultdict


# 修复的数据集类 - 确保数据加载稳定
class ShoeDataset(Dataset):
    def __init__(self, image_dir, metadata_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        # 构建图像路径映射
        self.image_paths = []
        self.valid_metadata = {}

        print("正在扫描图像文件...")
        missing_count = 0
        found_count = 0

        # 遍历所有元数据条目
        for img_name, attributes in self.metadata.items():
            found = False
            # 检查所有可能的子目录
            for style_dir in ['Boot', 'Sandal', 'Shoe', 'boots', 'sandals', 'shoes']:
                possible_paths = [
                    os.path.join(image_dir, style_dir, img_name),
                    os.path.join(image_dir, style_dir, img_name.lower()),
                    os.path.join(image_dir, style_dir, img_name.upper()),
                    os.path.join(image_dir, style_dir, img_name.replace(' ', '_')),
                    os.path.join(image_dir, style_dir, img_name.replace(' ', '')),
                ]

                # 添加扩展名变体
                base_name = os.path.splitext(img_name)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    possible_paths.append(os.path.join(image_dir, style_dir, base_name + ext))
                    possible_paths.append(os.path.join(image_dir, style_dir, base_name.lower() + ext))

                for full_path in possible_paths:
                    if os.path.exists(full_path):
                        self.image_paths.append(full_path)
                        self.valid_metadata[full_path] = attributes
                        found = True
                        found_count += 1
                        break

                if found:
                    break

            if not found:
                missing_count += 1
                if missing_count <= 5:
                    print(f"警告: 文件不存在，已跳过: {img_name}")

        print(f"找到图像数量: {found_count}/{len(self.metadata)}")
        print(f"缺失图像数量: {missing_count}")

        if len(self.image_paths) == 0:
            raise FileNotFoundError("没有找到任何有效图像文件！")

        # 构建词汇表
        self.colors = set()
        self.styles = set()

        for attributes in self.valid_metadata.values():
            self.colors.add(attributes['color'])
            self.styles.add(attributes['style'])

        self.colors = sorted(list(self.colors))
        self.styles = sorted(list(self.styles))

        self.color_to_idx = {color: idx for idx, color in enumerate(self.colors)}
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}

        print(f"可用的颜色 ({len(self.colors)}): {self.colors}")
        print(f"可用的款式 ({len(self.styles)}): {self.styles}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')

            # 确保图像是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            attributes = self.valid_metadata[img_path]
            color = attributes['color']
            style = attributes['style']

            color_idx = self.color_to_idx[color]
            style_idx = self.style_to_idx[style]

            # 创建条件向量
            condition = np.zeros(len(self.colors) + len(self.styles), dtype=np.float32)
            condition[color_idx] = 1.0
            condition[len(self.colors) + style_idx] = 1.0

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(condition, dtype=torch.float32)

        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            # 返回随机图像作为占位符
            if self.transform:
                placeholder = torch.randn(3, 64, 64)
            else:
                placeholder = Image.new('RGB', (64, 64), color='gray')
            condition = np.zeros(len(self.colors) + len(self.styles), dtype=np.float32)
            return placeholder, torch.tensor(condition, dtype=torch.float32)


# 统一的生成器架构
class UnifiedGenerator(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=10, img_channels=3):
        super(UnifiedGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.img_channels = img_channels

        # 投影层 - 将潜在向量转换为特征图
        self.projection = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        # 上采样层
        self.upsample = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 最终卷积层
            nn.Conv2d(32, img_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z, conditions):
        # 连接噪声和条件向量
        gen_input = torch.cat([z, conditions], dim=1)

        # 投影到特征空间
        out = self.projection(gen_input)
        out = out.view(-1, 512, 4, 4)

        # 上采样生成图像
        img = self.upsample(out)
        return img


# 统一的判别器架构
class UnifiedDiscriminator(nn.Module):
    def __init__(self, condition_dim=10, img_channels=3):
        super(UnifiedDiscriminator, self).__init__()

        self.condition_dim = condition_dim
        self.img_channels = img_channels

        # 下采样层
        self.downsample = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 + condition_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, conditions):
        # 提取特征
        features = self.downsample(img)
        features = features.view(features.size(0), -1)

        # 连接特征和条件向量
        combined = torch.cat([features, conditions], dim=1)

        # 分类
        validity = self.classifier(combined)
        return validity


# 优化的训练类
class OptimizedShoeGAN:
    def __init__(self, latent_dim=100, lr_g=0.0002, lr_d=0.0002, b1=0.5, b2=0.999):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.generator = None
        self.discriminator = None

        # 损失函数
        self.adversarial_loss = nn.BCELoss()

        # 优化器参数
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.b1 = b1
        self.b2 = b2

        # 训练历史
        self.g_losses = []
        self.d_losses = []

    def initialize_models(self, condition_dim):
        """初始化模型并确保架构匹配"""
        print(f"初始化模型 - 条件维度: {condition_dim}")

        self.generator = UnifiedGenerator(
            latent_dim=self.latent_dim,
            condition_dim=condition_dim
        ).to(self.device)

        self.discriminator = UnifiedDiscriminator(
            condition_dim=condition_dim
        ).to(self.device)

        # 优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.lr_g,
            betas=(self.b1, self.b2)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=(self.b1, self.b2)
        )

        # 学习率调度器
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=30, gamma=0.8)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=30, gamma=0.8)

        print("模型初始化完成")

    def train(self, dataloader, condition_dim, epochs=200, sample_interval=20):
        """训练模型"""
        self.initialize_models(condition_dim)

        os.makedirs("training_progress", exist_ok=True)

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0

            for i, (real_imgs, conditions) in enumerate(dataloader):
                batch_size = real_imgs.size(0)

                # 跳过太小的批次
                if batch_size < 2:
                    continue

                # 准备数据
                real_imgs = real_imgs.to(self.device)
                conditions = conditions.to(self.device)

                # 真实和假的标签
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)

                # ---------------------
                #  训练判别器
                # ---------------------
                self.optimizer_D.zero_grad()

                # 真实图像的损失
                real_loss = self.adversarial_loss(
                    self.discriminator(real_imgs, conditions), valid
                )

                # 生成图像的损失
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                gen_imgs = self.generator(z, conditions)
                fake_loss = self.adversarial_loss(
                    self.discriminator(gen_imgs.detach(), conditions), fake
                )

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # ---------------------
                #  训练生成器
                # ---------------------
                self.optimizer_G.zero_grad()

                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                gen_imgs = self.generator(z, conditions)

                g_loss = self.adversarial_loss(
                    self.discriminator(gen_imgs, conditions), valid
                )

                g_loss.backward()
                self.optimizer_G.step()

                # 记录损失
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                num_batches += 1

                if i % 50 == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

            # 更新学习率
            self.scheduler_G.step()
            self.scheduler_D.step()

            # 计算平均损失
            if num_batches > 0:
                avg_g_loss = epoch_g_loss / num_batches
                avg_d_loss = epoch_d_loss / num_batches
                self.g_losses.append(avg_g_loss)
                self.d_losses.append(avg_d_loss)

                print(f"Epoch {epoch} 完成 - D loss: {avg_d_loss:.6f}, G loss: {avg_g_loss:.6f}")

                # 保存样本和损失曲线
                if epoch % sample_interval == 0:
                    self._save_samples(epoch, conditions[:1])
                    self._plot_losses(epoch)

        # 保存最终模型
        self._save_models()

    def _save_samples(self, epoch, condition):
        """保存生成的样本图像"""
        with torch.no_grad():
            z = torch.randn(5, self.latent_dim, device=self.device)
            condition = condition.repeat(5, 1)
            gen_imgs = self.generator(z, condition)
            gen_imgs = 0.5 * gen_imgs + 0.5  # 反标准化

            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                img = gen_imgs[i].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig(f"training_progress/epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()

    def _plot_losses(self, epoch):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.title(f'Training Losses (Epoch {epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"training_progress/losses_epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _save_models(self):
        """保存模型和配置"""
        # 保存模型权重
        torch.save(self.generator.state_dict(), "generator_unified.pth")
        torch.save(self.discriminator.state_dict(), "discriminator_unified.pth")

        # 保存完整模型配置
        model_config = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'latent_dim': self.latent_dim,
            'condition_dim': self.generator.condition_dim,
            'img_channels': self.generator.img_channels
        }
        torch.save(model_config, "model_complete.pth")

        print("模型保存完成")


# 统一的设计器类
class UnifiedShoeDesigner:
    def __init__(self, model_path="model_complete.pth", vocab_path="vocab.json"):
        # 加载词汇表
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        # 加载模型配置
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.latent_dim = checkpoint['latent_dim']
            self.condition_dim = checkpoint['condition_dim']
            self.img_channels = checkpoint['img_channels']

            # 初始化模型
            self.generator = UnifiedGenerator(
                latent_dim=self.latent_dim,
                condition_dim=self.condition_dim,
                img_channels=self.img_channels
            )
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print("模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.generator.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)

    def design_shoe(self, color, style, num_variations=3):
        """设计鞋子"""
        # 创建条件向量
        condition = np.zeros(self.condition_dim, dtype=np.float32)

        if color in self.vocab['color_to_idx']:
            condition[self.vocab['color_to_idx'][color]] = 1.0
        else:
            print(f"警告: 颜色 '{color}' 不在词汇表中，使用默认颜色")

        if style in self.vocab['style_to_idx']:
            condition[len(self.vocab['colors']) + self.vocab['style_to_idx'][style]] = 1.0
        else:
            print(f"警告: 款式 '{style}' 不在词汇表中，使用默认款式")

        condition_tensor = torch.tensor(condition, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 生成多个变体并选择最好的
        best_image = None
        best_diversity = -1

        with torch.no_grad():
            for i in range(num_variations):
                z = torch.randn(1, self.latent_dim, device=self.device)
                gen_img = self.generator(z, condition_tensor)

                # 计算多样性得分（方差）
                diversity = torch.var(gen_img).item()

                if diversity > best_diversity:
                    best_diversity = diversity
                    best_image = gen_img

        # 后处理
        best_image = 0.5 * best_image + 0.5
        best_image = best_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        best_image = np.clip(best_image, 0, 1)

        return best_image


def main():
    """主训练函数"""
    print("开始优化训练...")

    # 配置路径
    image_dir = "D:/python/deep learn/archive/Shoe vs Sandal vs Boot Dataset"
    metadata_file = "shoe_metadata_improved_colors.json"

    # 数据增强变换
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    try:
        # 创建数据集
        dataset = ShoeDataset(
            image_dir=image_dir,
            metadata_file=metadata_file,
            transform=transform
        )

        print(f"数据集大小: {len(dataset)}")

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            drop_last=True  # 确保批次大小一致
        )

        # 计算条件维度
        condition_dim = len(dataset.colors) + len(dataset.styles)
        print(f"条件维度: {condition_dim}")

        # 保存词汇表
        vocab = {
            'colors': dataset.colors,
            'styles': dataset.styles,
            'color_to_idx': dataset.color_to_idx,
            'style_to_idx': dataset.style_to_idx
        }
        with open("vocab.json", 'w') as f:
            json.dump(vocab, f, indent=2)
        print("词汇表已保存")

        # 训练模型
        gan = OptimizedShoeGAN(latent_dim=100, lr_g=0.0002, lr_d=0.0002)
        gan.train(dataloader, condition_dim, epochs=100, sample_interval=10)

        print("训练完成！")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def test_generation():
    """测试生成功能"""
    print("测试图像生成...")

    if not os.path.exists("model_complete.pth") or not os.path.exists("vocab.json"):
        print("请先运行训练完成模型")
        return

    try:
        designer = UnifiedShoeDesigner()

        # 测试组合
        test_combinations = [
            ("red", "Shoe"),
            ("blue", "Sandal"),
            ("black", "Boot"),
            ("white", "Shoe"),
            ("brown", "Boot")
        ]

        # 生成图像
        os.makedirs("generated_shoes", exist_ok=True)

        fig, axes = plt.subplots(1, len(test_combinations), figsize=(20, 4))

        for idx, (color, style) in enumerate(test_combinations):
            try:
                shoe_image = designer.design_shoe(color, style)
                axes[idx].imshow(shoe_image)
                axes[idx].axis('off')
                axes[idx].set_title(f"{color} {style}", fontsize=12)

                # 单独保存每个图像
                plt.figure(figsize=(4, 4))
                plt.imshow(shoe_image)
                plt.axis('off')
                plt.savefig(f"generated_shoes/{color}_{style}.png",
                            dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

            except Exception as e:
                print(f"生成 {color} {style} 时出错: {e}")
                axes[idx].text(0.5, 0.5, f"Error\n{color} {style}",
                               ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("generated_shoes/all_combinations.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("测试完成！生成的图像保存在 generated_shoes 目录中")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("training_progress", exist_ok=True)
    os.makedirs("generated_shoes", exist_ok=True)

    # 运行训练
    main()

    # 测试生成
    test_generation()