# ViT-Attention-Visualization 🚀

![AI Focus Result](AI_Focus_Sister.jpg)

## 📌 项目简介
本项目基于 **Vision Transformer (ViT)** 架构，实现了图像特征提取的 **Attention Rollout** 可视化。通过量化分析深层网络中的注意力权重，直观展示了模型在处理非结构化复杂背景（如油纸伞、植被）时，如何通过全域感知锁定核心语义目标。

## 🛠️ 技术亮点
* **核心架构**: DeiT-Tiny (Data-efficient Image Transformers)
* **核心算法**: Attention Rollout (多层注意力矩阵递归聚合)
* **工程实践**: 
  - 针对 **RTX 3050Ti** 实现了完整的离线环境依赖链部署。
  - 解决了 SSL 证书受限环境下的模型权重加载与环境构建。
  - 实现了原图与 Heatmap 的像素级对齐可视化。

## 📊 实验结论
通过对人像样本的横向对比分析发现：
1. **浅层响应**: 模型对高频纹理（如发丝边缘、背景叶片）敏感。
2. **深层进化**: 随着 Block 层数加深，注意力有效收敛至人脸等高层语义区域，验证了 ViT 强大的 **Long-range Dependency** 建模能力。

## 🏃 运行环境
* Python 3.8 / PyTorch 2.4.0 / CUDA 11.8
* timm (PyTorch Image Models)
* OpenCV / Matplotlib

---

