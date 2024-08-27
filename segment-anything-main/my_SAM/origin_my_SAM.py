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
        csa1 = self.relu(self.bn(self.conv1(csa1)))
        vit2 = vit1 + csa1
        csa2 = self.relu(self.bn(self.conv2(vit2)))
        return csa2, vit2


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

        return mrm2_1, mrm2_2





class FeatureFusionDecoder(nn.Module):
    def __init__(self, out_channels):
        super(FeatureFusionDecoder, self).__init__()
        self.up_sample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_sample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up_sample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.up_sample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

        self.conv1 = nn.Conv2d(768, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(768, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(768, out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(768, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(768, out_channels, kernel_size=3, padding=1)
        # 在最后加一个1x1卷积，将768个通道压缩到类别数目
        self.final_conv1 = nn.Conv2d(768, 384, kernel_size=1)
        self.final_conv2 = nn.Conv2d(384, 192, kernel_size=1)
        self.final_conv3 = nn.Conv2d(192, 81, kernel_size=1)

    def forward(self, f3, f5, f7, f9, f11):
        x11 = self.conv1(f11)
        x9 = self.conv2(f9)
        x7 = self.conv3(f7)
        x5 = self.conv4(f5)
        x3 = self.conv5(f3)

        x9 = self.up_sample1(x9 + x11)
        x7 = self.up_sample2(x7 + x9)
        x5 = self.up_sample3(x5 + x7)
        x3 = self.up_sample4(x3 + x5)

        outputs = self.final_conv1(x3)
        outputs = self.final_conv2(outputs)
        outputs = self.final_conv3(outputs)
        return outputs

class TSSAM(nn.Module):
    def __init__(self, num_classes=81):
        super(TSSAM, self).__init__()
        self.image_encoder = ImageEncoderViT()  # 使用预训练的SAM图像编码器
        self.decoder = FeatureFusionDecoder(out_channels=num_classes)

    def forward(self, x):
        x = self.image_encoder.patch_embed(x)  # 首先进行Patch embedding
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed  # 添加绝对位置编码

        features = []
        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)
            if i in {2, 4, 6, 8, 10}:  # 在第 3, 5, 7, 9, 11 层提取特征
                features.append(x)

        f3, f5, f7, f9, f11 = features  # 从不同的尺度获取特征
        output = self.decoder(f3, f5, f7, f9, f11)  # 解码生成分割结果
        return output

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

    # 初始化模型、损失函数、优化器和混合精度的Scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TSSAM(num_classes=81).to(device)
    pretrained_weights = torch.load("D:/SAM/segment-anything-main/sam_vit_b_01ec64.pth")

    # 过滤掉不属于 image_encoder 的权重
    image_encoder_weights = {k.replace("image_encoder.", ""): v for k, v in pretrained_weights.items() if
                             k.startswith("image_encoder.")}
    # 加载权重到 image_encoder
    model.image_encoder.load_state_dict(image_encoder_weights, strict=False)

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
