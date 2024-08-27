import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import os
import torch.optim as optim
import torch.nn as nn
from segment_anything.modeling.image_encoder import ImageEncoderViT
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
import time

from torchvision import transforms
import xml.etree.ElementTree as ET

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 调整图像大小为1024x1024
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

class VOCSegmentationDataset(Dataset):
    def __init__(self, root, image_set='train', transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, "JPEGImages")
        self.mask_dir = os.path.join(root, "SegmentationClass")
        self.image_set = image_set

        image_set_path = os.path.join(root, "ImageSets", "Segmentation", f"{image_set}.txt")
        with open(image_set_path, "r") as f:
            self.image_list = f.read().splitlines()

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_name}.png")

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P')

        if self.transform is not None:
            img = self.transform(img)
            mask = transforms.functional.resize(mask, (1024, 1024), interpolation=Image.NEAREST)
            mask = torch.tensor(np.array(mask), dtype=torch.long)  # 转为long类型用于交叉熵损失

        return img, mask

    def __len__(self):
        return len(self.image_list)


class ConvolutionalSideAdapter(nn.Module):
    def __init__(self, in_channels1, out_channels1, out_channels2):
        super(ConvolutionalSideAdapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels1, out_channels=out_channels1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.bn2 = nn.BatchNorm2d(out_channels2)
        self.relu = nn.ReLU()


    def forward(self, csa1, vit1):
        csa1 = csa1.permute(0, 3, 1, 2) # 转成B C H W 2 768 64 64
        vit1 = vit1.permute(0, 3, 1, 2) # 2 768 64 64
        csa1 = self.relu(self.bn1(self.conv1(csa1)))
        vit2 = vit1 + csa1
        csa2 = self.relu(self.bn2(self.conv2(vit2)))
        return csa2.permute(0, 2, 3, 1), vit2.permute(0, 2, 3, 1) # 转回 B H W C


class MultiScaleRefineModule(nn.Module):
    def __init__(self, in_channels1, out_channels1):
        super(MultiScaleRefineModule, self).__init__()
        # 1x1卷积模块用于压缩vit特征
        self.conv1 = nn.Conv2d(in_channels=in_channels1, out_channels=out_channels1, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels1)
        self.relu = nn.ReLU()

        # 反卷积模块用于上采样
        self.deconv1 = nn.ConvTranspose2d(in_channels=out_channels1, out_channels=out_channels1, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=out_channels1, out_channels=out_channels1, kernel_size=4, stride=4)

        # 门控单元
        self.gate1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.gate2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Tanh()
        )

    def forward(self, vit1, mrm1=None, mrm2=None):
        vit1 = vit1.permute(0, 3, 1, 2) # 转成B C H W
        if mrm1 is not None:
            mrm1 = mrm1.permute(0, 3, 1, 2) # 转成B C H W
            mrm2 = mrm2.permute(0, 3, 1, 2) # 转成B C H W

        # 处理vit特征
        vit1 = self.relu(self.bn(self.conv1(vit1)))

        # 上采样获取高分辨率特征
        mrm_high_res1 = self.relu(self.bn(self.deconv1(vit1)))
        mrm_high_res2 = self.relu(self.bn(self.deconv2(vit1)))

        # 初始化mrm1,mrm2
        if mrm1 is None:
            mrm1 = torch.zeros_like(mrm_high_res1)
        if mrm2 is None:
            mrm2 = torch.zeros_like(mrm_high_res2)

        # 应用门控单元
        gate_out1 = self.gate1(mrm_high_res1)
        gate_out2 = self.gate2(mrm_high_res2)

        # 计算融合特征
        mrm2_1 = gate_out1 * mrm_high_res1 + mrm1
        mrm2_2 = gate_out2 * mrm_high_res2 + mrm2

        return mrm2_1.permute(0, 2, 3, 1), mrm2_2.permute(0, 2, 3, 1) # 转回 B H W C




class TSSAM(nn.Module):
    def __init__(self, image_encoder, num_class=8, in_channels=768, out_channels=256):
        super(TSSAM, self).__init__()
        self.image_encoder = image_encoder  # 使用预训练的SAM图像编码器，修改为256

        # 线性投影层，将原始图像特征投影到256维,256 -> 128
        self.linear_projection = nn.Conv2d(3, 128, kernel_size=1)  # 3表示输入的RGB通道

        self.t = ConvolutionalSideAdapter(128, 768, 128)

        # 创建CSA和MRM模块，每个ViT Block后一个
        self.csa_modules = nn.ModuleList([
            ConvolutionalSideAdapter(128, 768, 128) for _ in range(14)
        ])
        self.mrm_modules = nn.ModuleList([
            MultiScaleRefineModule(768, 128) for _ in range(12)
        ])
        self.mrm2 = MultiScaleRefineModule(256, 128)

        self.neck = image_encoder.neck

        self.conv3 = ConvolutionalSideAdapter(768, 256, 256)

        # FFD部分
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 768, kernel_size=3, padding=1)
        self.stage1_conv = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.stage1_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.stage2_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(128, num_class, kernel_size=1)  # 最终输出num_class通道的掩码

    def forward(self, x):
        # 原始图像
        origin_x = x.to(device)

        # Patch embedding
        x = self.image_encoder.patch_embed(x) # B H W C
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed  # 添加绝对位置编码

        # 首先将图像从 1024x1024 调整为 64x64
        y = F.avg_pool2d(origin_x, kernel_size=16)
        # 线性投影
        proj_x = self.linear_projection(y).permute(0, 2, 3, 1)  # B C H W -> B H W C

        # 第一次CSA操作使用proj_x和x
        csa_out, x = self.t(proj_x, x)
        mrm_outputs = [None, None]

        x = self.image_encoder.blocks[0](x) # 第一个Vit Block

        # 依次通过剩余11个ViT Block，并在每个Block后应用CSA和MRM
        for i in range(1,12):
            x = self.image_encoder.blocks[i](x) # x 的shape为 B H W C
            csa_out, x = self.csa_modules[i](csa_out, x)  # 使用上一个CSA的输出和当前ViT Block的输出
            mrm_outputs = self.mrm_modules[i](x, *mrm_outputs)

        csa_out = csa_out.permute(0, 3, 1, 2)
        csa_out = self.conv2(csa_out)
        # 1 768 64 64
        # ViT Neck
        neck_out = self.neck(csa_out)  # 维度变为 [B, C, H, W]
        # 1 256 64 64
        csa_out, x = self.conv3(csa_out.permute(0, 2, 3, 1), neck_out.permute(0, 2, 3, 1))
        mrm_outputs = self.mrm2(x, *mrm_outputs)

        # FFD部分
        F1_mrm, F2_mrm = mrm_outputs
        F1_mrm = F1_mrm.permute(0, 3, 1, 2) # 维度变为 [B, C, H, W], 适应后续的卷积
        F2_mrm = F2_mrm.permute(0, 3, 1, 2) # 维度变为 [B, C, H, W]
        csa_out = csa_out.permute(0, 3, 1, 2) # 维度变为 [B, C, H, W]

        F1_mrm = self.conv(F1_mrm)
        F2_mrm = self.conv(F2_mrm)

        # 池化操作提取关键特征
        F1_key_mrm = F.adaptive_avg_pool2d(F1_mrm, (F1_mrm.size(2) // 2, F1_mrm.size(3) // 2)) + \
                     F.adaptive_max_pool2d(F1_mrm, (F1_mrm.size(2) // 2, F1_mrm.size(3) // 2))
        # H/8 -> H/16, W/8 -> W/16
        F2_key_mrm = F.adaptive_avg_pool2d(F2_mrm, (F2_mrm.size(2) // 2, F2_mrm.size(3) // 2)) + \
                     F.adaptive_max_pool2d(F2_mrm, (F2_mrm.size(2) // 2, F2_mrm.size(3) // 2))
        # H/4 -> H/8, W/4 -> W/8

        # 第一个阶段的融合
        Fin_csa = self.conv1x1(csa_out)  # csa_out 已经是 [B, C, H, W]，不需要 permute
        F_stage1_csa = torch.cat([Fin_csa, F1_key_mrm], dim=1) # C + C = 256 + 256 = 512
        F_stage1_csa = self.stage1_conv(F_stage1_csa) # C -> 256
        F_stage1_csa = F.interpolate(F_stage1_csa, scale_factor=2, mode='bilinear', align_corners=False)
        # [B, C, H/8, W/8]

        # 第二个阶段的融合
        F_stage2_csa = torch.cat([F_stage1_csa, F1_mrm], dim=1)  # F1_mrm 已经是 [B, C, H, W]
        F_stage2_csa = self.stage2_conv(F_stage2_csa)
        # [B, C, H/8, W/8]

        # 第三个阶段的融合
        F_stage3_csa = torch.cat([F_stage2_csa, F2_key_mrm], dim=1)  # C + C = 256 + 256 = 512
        F_stage3_csa = self.stage1_conv2(F_stage3_csa)  # C -> 256
        F_stage3_csa = F.interpolate(F_stage3_csa, scale_factor=2, mode='bilinear', align_corners=False)
        # [B, C, H/4, W/4]

        # 第四个阶段的融合
        F_stage4_csa = torch.cat([F_stage3_csa, F2_mrm], dim=1)
        F_stage4_csa = self.stage2_conv(F_stage4_csa)
        # [B, C, H/4, W/4] C = 256

        # [B, C, H/4, W/4]如何变成[B, C, H, W]，原文中并未提及，这里我使用F_stage4_csa上采样后与原始图像拼接，再卷积
        # 进行最终的上采样，将 F_stage4_csa 上采样到与 proj_x 相同的分辨率
        F_stage4_csa = F.interpolate(F_stage4_csa, size=(1024, 1024), mode='bilinear',
                                     align_corners=False)
        # [B, C, H, W]

        # # 拼接 F_stage4_csa 和 proj_x 在通道维度上 (dim=1)
        # final_features = torch.cat([F_stage4_csa, origin_x], dim=1)

        # 使用卷积层将拼接后的特征图生成最终的掩码
        final_mask = self.final_conv(F_stage4_csa)
        # C=512 -> num_class

        return final_mask




def custom_collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks

if __name__ == "__main__":
    batch_size = 1
    root_dir = "F:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"

    # 创建数据加载器
    train_dataset = VOCSegmentationDataset(root=root_dir, image_set='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=custom_collate_fn)

    val_dataset = VOCSegmentationDataset(root=root_dir, image_set='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                            collate_fn=custom_collate_fn)

    # 训练TS-SAM模型
    from torch.cuda.amp import GradScaler, autocast

    # 初始化模型、损失函数、优化器和混合精度的Scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_encoder = ImageEncoderViT().to(device)  # 实例化

    pretrained_weights = torch.load("F:\SAM\sam_vit_b_01ec64.pth")

    # 过滤掉不属于 image_encoder 的权重
    image_encoder_weights = {k.replace("image_encoder.", ""): v for k, v in pretrained_weights.items() if
                             k.startswith("image_encoder.")}
    # 加载权重到 image_encoder
    image_encoder.load_state_dict(image_encoder_weights, strict=False)

    model = TSSAM(image_encoder, num_class=21).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    # 冻结 image_encoder 的权重
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # 现在只有新增加的模块会被训练
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)



    # 初始化梯度缩放器
    scaler = GradScaler()
    num_epochs = 1
    best_val_loss = float('inf')  # 初始化最佳验证损失

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            # start_time = time.time()
            images = images.to(device).float()
            masks = masks.to(device)

            optimizer.zero_grad()

            # 前向传播和计算损失
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播和梯度更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            # end_time = time.time()

            # print(f"Batch {i} processing time: {end_time - start_time:.4f} seconds")

            if i % 100 == 99:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device).float()
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with loss: {best_val_loss:.4f}')

    print("Training complete.")




