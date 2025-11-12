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
import torch.nn.functional as F

# 设置GPU内存优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

# 轻量级残差块 - 提高细节表现
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=False)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        elif upsample:
            self.upsample_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            self.upsample_bn = nn.BatchNorm2d(out_channels)
        elif in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None

    def forward(self, x):
        identity = x
        
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            identity = F.interpolate(identity, scale_factor=2, mode='nearest')
        
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        
        if self.downsample:
            identity = self.downsample_bn(self.downsample_conv(F.avg_pool2d(x, 2)))
        elif self.upsample and self.upsample_conv is not None:
            identity = self.upsample_bn(self.upsample_conv(identity))
        elif self.skip_conv is not None:
            identity = self.skip_bn(self.skip_conv(identity))
        
        out += identity
        return F.leaky_relu(out, 0.2)

# 轻量级注意力机制
class LightweightAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(LightweightAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ca = self.channel_attention(x)
        return x * ca

# 修复的生成器 - 确保输出64x64图像
class EnhancedGenerator(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=10, img_channels=3, output_size=64):
        super(EnhancedGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.img_channels = img_channels
        self.output_size = output_size

        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 初始卷积
        self.initial_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 计算需要的上采样次数
        self.upsample_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(256, 128),
            LightweightAttention(128),
            
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(128, 64),
            LightweightAttention(64),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(64, 32),
            LightweightAttention(32),
            
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(32, 16),
            LightweightAttention(16),
        )

        # 最终卷积层
        self.final_convs = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, img_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z, conditions):
        gen_input = torch.cat([z, conditions], dim=1)
        out = self.projection(gen_input)
        out = out.view(-1, 256, 4, 4)
        out = self.initial_conv(out)
        out = self.upsample_layers(out)
        img = self.final_convs(out)
        return img

# 修复的判别器 - 确保处理64x64输入
class EnhancedDiscriminator(nn.Module):
    def __init__(self, condition_dim=10, img_channels=3, input_size=64):
        super(EnhancedDiscriminator, self).__init__()
        
        self.condition_dim = condition_dim
        self.img_channels = img_channels
        self.input_size = input_size

        # 多尺度特征提取
        self.feature_extractor = nn.Sequential(
            # 输入: 64x64
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            
            # 第二层
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            LightweightAttention(128),
            
            # 第三层
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            
            # 第四层
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8x8 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            
            # 残差块提取更丰富特征
            ResidualBlock(512, 512),
            LightweightAttention(512),
        )

        # 计算特征图大小
        self.feature_size = input_size // 16  # 经过4次下采样: 64->32->16->8->4
        self.feature_channels = 512

        # 全局平均池化 + 最大池化
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_channels * 2 + condition_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, conditions):
        # 检查输入尺寸
        if img.size(2) != self.input_size or img.size(3) != self.input_size:
            # 调整输入尺寸以匹配期望的输入大小
            img = F.interpolate(img, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        features = self.feature_extractor(img)
        
        # 多尺度特征聚合
        avg_features = self.global_pool[0](features)
        max_features = self.global_pool[1](features)
        pooled_features = torch.cat([avg_features, max_features], dim=1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        combined = torch.cat([pooled_features, conditions], dim=1)
        validity = self.classifier(combined)
        return validity

# 数据集类保持不变
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

        for img_name, attributes in self.metadata.items():
            found = False
            for style_dir in ['Boot', 'Sandal', 'Shoe', 'boots', 'sandals', 'shoes']:
                possible_paths = [
                    os.path.join(image_dir, style_dir, img_name),
                    os.path.join(image_dir, style_dir, img_name.lower()),
                    os.path.join(image_dir, style_dir, img_name.upper()),
                    os.path.join(image_dir, style_dir, img_name.replace(' ', '_')),
                    os.path.join(image_dir, style_dir, img_name.replace(' ', '')),
                ]

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

            if image.mode != 'RGB':
                image = image.convert('RGB')

            attributes = self.valid_metadata[img_path]
            color = attributes['color']
            style = attributes['style']

            color_idx = self.color_to_idx[color]
            style_idx = self.style_to_idx[style]

            condition = np.zeros(len(self.colors) + len(self.styles), dtype=np.float32)
            condition[color_idx] = 1.0
            condition[len(self.colors) + style_idx] = 1.0

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(condition, dtype=torch.float32)

        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            if self.transform:
                placeholder = torch.randn(3, 64, 64)  # 确保是64x64
            else:
                placeholder = Image.new('RGB', (64, 64), color='gray')
            condition = np.zeros(len(self.colors) + len(self.styles), dtype=np.float32)
            return placeholder, torch.tensor(condition, dtype=torch.float32)

# 修复的训练类 - 简化梯度惩罚
class EnhancedShoeGAN:
    def __init__(self, latent_dim=64, lr_g=0.0001, lr_d=0.0004, b1=0.5, b2=0.999):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 设置GPU内存限制
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)

        self.generator = None
        self.discriminator = None

        # 使用标准BCE损失提高稳定性
        self.adversarial_loss = nn.BCELoss()

        self.lr_g = lr_g
        self.lr_d = lr_d
        self.b1 = b1
        self.b2 = b2

        self.g_losses = []
        self.d_losses = []

    def initialize_models(self, condition_dim):
        """初始化改进的模型"""
        print(f"初始化改进模型 - 条件维度: {condition_dim}")

        self.generator = EnhancedGenerator(
            latent_dim=self.latent_dim,
            condition_dim=condition_dim,
            output_size=64  # 明确指定输出尺寸
        ).to(self.device)

        self.discriminator = EnhancedDiscriminator(
            condition_dim=condition_dim,
            input_size=64  # 明确指定输入尺寸
        ).to(self.device)

        # 使用Adam优化器
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

        # 学习率调度
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=30, gamma=0.8)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=30, gamma=0.8)

        print("改进模型初始化完成")

    def train(self, dataloader, condition_dim, epochs=100, sample_interval=10):
        """简化的训练循环，避免梯度惩罚问题"""
        self.initialize_models(condition_dim)

        os.makedirs("training_progress_enhanced", exist_ok=True)

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0

            for i, (real_imgs, conditions) in enumerate(dataloader):
                batch_size = real_imgs.size(0)

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
                
                # 检查尺寸是否匹配
                if gen_imgs.size() != real_imgs.size():
                    print(f"尺寸不匹配: 生成图像 {gen_imgs.size()}, 真实图像 {real_imgs.size()}")
                    # 调整生成图像尺寸以匹配真实图像
                    gen_imgs = F.interpolate(gen_imgs, size=real_imgs.shape[2:], mode='bilinear', align_corners=False)
                
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
                
                # 再次检查尺寸
                if gen_imgs.size() != real_imgs.size():
                    gen_imgs = F.interpolate(gen_imgs, size=real_imgs.shape[2:], mode='bilinear', align_corners=False)

                g_loss = self.adversarial_loss(
                    self.discriminator(gen_imgs, conditions), valid
                )

                g_loss.backward()
                self.optimizer_G.step()

                # 记录损失
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                num_batches += 1

                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if i % 20 == 0:
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
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                img = gen_imgs[i].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig(f"training_progress_enhanced/epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
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
        plt.savefig(f"training_progress_enhanced/losses_epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _save_models(self):
        """保存模型和配置"""
        torch.save(self.generator.state_dict(), "generator_enhanced.pth")
        torch.save(self.discriminator.state_dict(), "discriminator_enhanced.pth")

        model_config = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'latent_dim': self.latent_dim,
            'condition_dim': self.generator.condition_dim,
            'img_channels': self.generator.img_channels
        }
        torch.save(model_config, "model_complete_enhanced.pth")

        print("改进模型保存完成")

# 改进的设计器类
class EnhancedShoeDesigner:
    def __init__(self, model_path="model_complete_enhanced.pth", vocab_path="vocab.json"):
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.latent_dim = checkpoint['latent_dim']
            self.condition_dim = checkpoint['condition_dim']
            self.img_channels = checkpoint['img_channels']

            self.generator = EnhancedGenerator(
                latent_dim=self.latent_dim,
                condition_dim=self.condition_dim,
                img_channels=self.img_channels,
                output_size=64
            )
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print("改进模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.generator.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)

    def design_shoe(self, color, style, num_variations=10):
        """设计鞋子 - 使用更多变体选择最佳结果"""
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

        # 生成更多变体并选择最佳
        best_image = None
        best_score = -1

        with torch.no_grad():
            for i in range(num_variations):
                z = torch.randn(1, self.latent_dim, device=self.device)
                gen_img = self.generator(z, condition_tensor)

                # 更精细的质量评估
                # 1. 清晰度评估（梯度幅值）
                gradient_x = torch.abs(gen_img[:, :, :, 1:] - gen_img[:, :, :, :-1])
                gradient_y = torch.abs(gen_img[:, :, 1:, :] - gen_img[:, :, :-1, :])
                sharpness = (gradient_x.mean() + gradient_y.mean()).item()

                # 2. 对比度评估
                contrast = torch.std(gen_img).item()

                # 3. 细节丰富度（高频成分）
                laplacian = torch.abs(gen_img[:, :, 1:, 1:] - gen_img[:, :, :-1, :-1])
                detail_richness = laplacian.mean().item()

                # 综合得分
                score = sharpness * 0.5 + contrast * 0.3 + detail_richness * 0.2

                if score > best_score:
                    best_score = score
                    best_image = gen_img

        # 后处理 - 增强对比度
        best_image = 0.5 * best_image + 0.5
        best_image = best_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # 简单的对比度增强
        best_image = np.clip(best_image * 1.1 - 0.05, 0, 1)  # 轻微增强对比度
        
        return best_image

def main():
    """修复的主训练函数"""
    print("开始修复训练 - 解决尺寸不匹配问题...")

    # 配置路径
    image_dir = "D:\DeepLearning\\archive\Shoe vs Sandal vs Boot Dataset"
    metadata_file = "shoe_metadata_improved_colors.json"

    # 使用64x64图像
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 固定为64x64
        transforms.RandomCrop((64, 64)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

        # 使用小批次确保内存安全
        dataloader = DataLoader(
            dataset,
            batch_size=4,  # 减小批次大小确保内存安全
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=False
        )

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

        # 使用修复的GAN训练
        gan = EnhancedShoeGAN(latent_dim=64, lr_g=0.0001, lr_d=0.0004)
        gan.train(dataloader, condition_dim, epochs=100, sample_interval=5)

        print("训练完成！")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def test_generation():
    """测试生成功能"""
    print("测试修复的图像生成...")

    if not os.path.exists("model_complete_enhanced.pth") or not os.path.exists("vocab.json"):
        print("请先运行训练完成模型")
        return

    try:
        designer = EnhancedShoeDesigner()

        test_combinations = [
            ("red", "Shoe"),
            ("blue", "Sandal"), 
            ("black", "Boot"),
            ("white", "Shoe"),
            ("brown", "Boot")
        ]

        os.makedirs("generated_shoes_enhanced", exist_ok=True)

        fig, axes = plt.subplots(1, len(test_combinations), figsize=(20, 4))

        for idx, (color, style) in enumerate(test_combinations):
            try:
                shoe_image = designer.design_shoe(color, style, num_variations=10)
                axes[idx].imshow(shoe_image)
                axes[idx].axis('off')
                axes[idx].set_title(f"{color} {style}", fontsize=12)

                plt.figure(figsize=(6, 6))
                plt.imshow(shoe_image)
                plt.axis('off')
                plt.savefig(f"generated_shoes_enhanced/{color}_{style}.png",
                            dpi=200, bbox_inches='tight', pad_inches=0)
                plt.close()

            except Exception as e:
                print(f"生成 {color} {style} 时出错: {e}")
                axes[idx].text(0.5, 0.5, f"Error\n{color} {style}",
                               ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("generated_shoes_enhanced/all_combinations.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("测试完成！生成的图像保存在 generated_shoes_enhanced 目录中")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("training_progress_enhanced", exist_ok=True)
    os.makedirs("generated_shoes_enhanced", exist_ok=True)

    # 运行训练
    main()

    # 测试生成
    test_generation()