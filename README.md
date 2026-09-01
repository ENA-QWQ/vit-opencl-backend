# ViT-OpenCL-Backend

An experimental low-level implementation of Vision Transformer based on OpenCL.

This project is not intended as a general-purpose library. Not suitable for production use. For learning and reference purposes only.

---

## Features

### Forward Pass
- Patch Embedding
- Positional Encoding
- Multi-Head Self-Attention
- FFN + GELU
- LayerNorm
- Classification Head

### Backward Pass
- Gradient Computation
- Gradient Clipping
- AdamW Optimizer Update

## References

Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

## License

[MIT License](LICENSE)

---

# ViT-OpenCL-Backend
基于 OpenCL 的 Vision Transformer 实验性底层实现

本项目不作为一个通用库提供。不适合用于生产环境，仅供原理学习和参考。

---

## 功能列表
### 前向传播
- 图像块嵌入
- 位置编码
- 多头自注意力
- FFN + GELU
- LayerNorm
- 分类头

### 反向传播
- 梯度计算
- 梯度裁剪
- AdamW 优化器更新

## 参考文献
Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

## 许可证
[MIT License](LICENSE)