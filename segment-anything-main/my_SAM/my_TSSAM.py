import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import os
import torch.optim as optim
import torch.nn as nn
from segment_anything.modeling.image_encoder import ImageEncoderViT
import torch.nn.functional as F

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 调整图像大小为1024x1024
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

class COCOSegmentationDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        masks = [coco.annToMask(ann) for ann in anns]

        # 创建一个空白的mask，尺寸与图片一致
        mask = torch.zeros((img.size[1], img.size[0]), dtype=torch.uint8)
        for m in masks:
            mask = mask | torch.tensor(m, dtype=torch.uint8)

        # 统一对图像和掩码进行预处理
        if self.transform is not None:
            img = self.transform(img)
            mask = mask.unsqueeze(0)  # 添加一个通道维度 (1, H, W)
            mask = transforms.functional.resize(mask, (1024, 1024), interpolation=Image.NEAREST)
            mask = mask.squeeze(0)  # 去掉添加的通道维度 (H, W)

        return img, mask

    def __len__(self):
        return len(self.ids)


class ConvolutionalSideAdapter(nn.Module):
    def __init__(self, in_channels1, out_channels1, out_channels2):
        super(ConvolutionalSideAdapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels1)
        self.relu = nn.ReLU()

    def forward(self, csa1, vit1):
        csa1 = csa1.permute(0, 3, 1, 2) # 转成B C H W
        vit1 = vit1.permute(0, 3, 1, 2)
        csa1 = self.relu(self.bn(self.conv1(csa1)))
        vit2 = vit1 + csa1
        csa2 = self.relu(self.bn(self.conv2(vit2)))
        return csa2.permute(0, 2, 3, 1), vit2.permute(0, 2, 3, 1) # 转回 B H W C


class MultiScaleRefineModule(nn.Module):
    def __init__(self, in_channels1, out_channels1):
        super(MultiScaleRefineModule, self).__init__()
        # 1x1卷积模块用于压缩vit特征
        self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels1)
        self.relu = nn.ReLU()

        # 反卷积模块用于上采样
        self.deconv1 = nn.ConvTranspose2d(out_channels1, out_channels1, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(out_channels1, out_channels1, kernel_size=4, stride=4)

        # 门控单元
        self.gate1 = nn.Sequential(
            nn.Linear(out_channels1, out_channels1),
            nn.ReLU(),
            nn.Linear(out_channels1, out_channels1),
            nn.Tanh()
        )

        self.gate2 = nn.Sequential(
            nn.Linear(out_channels1, out_channels1),
            nn.ReLU(),
            nn.Linear(out_channels1, out_channels1),
            nn.Tanh()
        )

    def forward(self, vit1, mrm1=None, mrm2=None):
        vit1 = vit1.permute(0, 3, 1, 2) # 转成B C H W
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
        self.image_encoder = image_encoder  # 使用预训练的SAM图像编码器

        # 线性投影层，将原始图像特征投影到256维
        self.linear_projection = nn.Conv2d(3, 256, kernel_size=1)  # 3表示输入的RGB通道

        # 创建CSA和MRM模块，每个ViT Block后一个
        self.csa_modules = nn.ModuleList([
            ConvolutionalSideAdapter(256, 768, 768) for _ in range(14)
        ])
        self.mrm_modules = nn.ModuleList([
            MultiScaleRefineModule(768, 256) for _ in range(13)
        ])

        self.neck = image_encoder.neck

        # FFD部分
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.stage1_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.stage2_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(512, num_class, kernel_size=1)  # 最终输出num_class通道的掩码

    def forward(self, x):
        # 线性投影
        proj_x = self.linear_projection(x).permute(0, 2, 3, 1) # B C H W -> B H W C

        # Patch embedding
        x = self.image_encoder.patch_embed(x) # B H W C
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed  # 添加绝对位置编码

        # 第一次CSA操作使用proj_x和x
        csa_out, x = self.csa_modules[12](proj_x, x)
        mrm_outputs = [None, None]

        x = self.image_encoder.blocks[0](x) # 第一个Vit Block

        # 依次通过剩余11个ViT Block，并在每个Block后应用CSA和MRM
        for i in range(1,12):
            x = self.image_encoder.blocks[i](x) # x 的shape为 B H W C
            csa_out, x = self.csa_modules[i](csa_out, x)  # 使用上一个CSA的输出和当前ViT Block的输出
            mrm_outputs = self.mrm_modules[i](x, *mrm_outputs)

        # ViT Neck
        neck_out = self.neck(csa_out.permute(0, 3, 1, 2))  # 维度变为 [B, C, H, W]
        neck_out = neck_out.permute(0, 2, 3, 1)  # 维度变为 [B, H, W, C]以适应 CSA 的输入要求
        csa_out, x = ConvolutionalSideAdapter(256, 256, 256)(csa_out, neck_out)
        mrm_outputs = self.mrm_modules[-1](x, *mrm_outputs)

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
        F_stage3_csa = self.stage1_conv(F_stage3_csa)  # C -> 256
        F_stage3_csa = F.interpolate(F_stage3_csa, scale_factor=2, mode='bilinear', align_corners=False)
        # [B, C, H/4, W/4]

        # 第四个阶段的融合
        F_stage4_csa = torch.cat([F_stage3_csa, F2_mrm], dim=1)
        F_stage4_csa = self.stage2_conv(F_stage4_csa)
        # [B, C, H/4, W/4] C = 256

        # [B, C, H/4, W/4]如何变成[B, C, H, W]，原文中并未提及，这里我使用F_stage4_csa上采样后与proj_x拼接，再卷积
        # 进行最终的上采样，将 F_stage4_csa 上采样到与 proj_x 相同的分辨率
        F_stage4_csa = F.interpolate(F_stage4_csa, size=(proj_x.size(2), proj_x.size(3)), mode='bilinear',
                                     align_corners=False)
        # [B, C, H, W]

        # 拼接 F_stage4_csa 和 proj_x 在通道维度上 (dim=1)
        final_features = torch.cat([F_stage4_csa, proj_x], dim=1)

        # 使用卷积层将拼接后的特征图生成最终的掩码
        final_mask = self.final_conv(final_features)
        # C=512 -> num_class

        return final_mask




def custom_collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks

if __name__ == "__main__":
    batch_size = 2
    # 创建数据加载器
    train_dataset = COCOSegmentationDataset(root="D:/coco2017/train2017",
                                            annFile="D:/coco2017/annotations/instances_train2017.json",
                                            transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)

    val_dataset = COCOSegmentationDataset(root="D:/coco2017/test2017",
                                          annFile="D:/coco2017/annotations/instances_val2017.json",
                                          transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)

    # 训练TS-SAM模型
    from torch.cuda.amp import GradScaler, autocast

    image_encoder = ImageEncoderViT() # 实例化

    # 初始化模型、损失函数、优化器和混合精度的Scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_weights = torch.load("F:\SAM\sam_vit_b_01ec64.pth")

    # 过滤掉不属于 image_encoder 的权重
    image_encoder_weights = {k.replace("image_encoder.", ""): v for k, v in pretrained_weights.items() if
                             k.startswith("image_encoder.")}
    # 加载权重到 image_encoder
    image_encoder.load_state_dict(image_encoder_weights, strict=False)

    model = TSSAM(image_encoder, num_class=81).to(device)

    criterion = nn.CrossEntropyLoss()
    # 冻结 image_encoder 的权重
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # 现在只有新增加的模块会被训练
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    scaler = GradScaler()
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # 使用 autocast 进行混合精度训练
            with autocast():
                outputs = model(images)
                print(outputs.shape)
                input()
                loss = criterion(outputs, masks)

            # Scaler 处理梯度计算
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % 100 == 99:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss / len(val_loader):.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'ts_sam_coco.pth')
