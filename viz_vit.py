import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os
import timm

# 1. 自动检测环境并加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model('deit_tiny_patch16_224', pretrained=True).to(device)
model.eval()

def visualize_attention(img_path, output_path):
    # 2. 图像预处理 (224x224 是 DeiT 的标准重量)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 3. 提取注意力矩阵 (DeiT 特有的 Block 结构)
    # 我们取最后一层的注意力来展示最终的语义聚焦
    attentions = []
    def hook_fn(module, input, output):
        # timm 的 DeiT 结构中，output 是注意力权重
        attentions.append(output)

    # 挂载钩子到最后一层的自注意力模块
    handle = model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(input_tensor)
    handle.remove()

    # 4. Attention Rollout 简化计算 (展示最后一层)
    attn = attentions[0] # shape: [1, num_heads, 197, 197]
    attn_map = attn.mean(dim=1).squeeze(0) # 平均所有头
    cls_weight = attn_map[0, 1:].reshape(14, 14).cpu().numpy() # 取 Class Token 对其他 Patch 的权重
    
    # 插值缩放回原图大小
    cls_weight = (cls_weight - cls_weight.min()) / (cls_weight.max() - cls_weight.min())
    
    # 5. 画图并保存
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.resize((224, 224)))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cls_weight, cmap='jet', interpolation='bicubic')
    plt.title("DeiT Attention Map")
    plt.axis('off')

    plt.savefig(output_path)
    plt.close()
    print(f"✅ 处理完成: {img_path} -> {output_path}")

# --- 批量处理逻辑 ---
input_dir = "samples"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for f in os.listdir(input_dir):
    if f.endswith(('.jpg', '.png', '.jpeg')):
        visualize_attention(os.path.join(input_dir, f), os.path.join(output_dir, f"vis_{f}"))