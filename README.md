# learn-triton

个人学习 [Triton](https://triton-lang.org/)（GPU 内核编程语言）的笔记与实验仓库。

## 环境要求

- **Python**：3.10+（建议 3.10 或 3.11）
- **GPU**：NVIDIA GPU，且本机已安装与驱动匹配的 **CUDA Toolkit**（具体版本以你安装的 PyTorch / Triton 轮子为准）
- **操作系统**：Linux 最常见；macOS 上通常无 CUDA，多用于读代码；Windows 需自行确认 PyTorch + CUDA 组合

## 快速开始

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install torch triton
```

若你使用 `requirements.txt`（后续可自行添加），可改为：

```bash
pip install -r requirements.txt
```

## 仓库结构（建议）

后续可按主题建目录，例如：

| 目录 | 说明 |
|------|------|
| `examples/` | 官方文档风格的最小可运行样例 |
| `notes/` | 学习笔记（可选） |
| `kernels/` | 自己写的 kernel 与 benchmark |

当前仓库为空时，可先在本文件记录学习进度与命令，再逐步补代码。

## 学习资源

- [Triton 官方文档](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton GitHub](https://github.com/triton-lang/triton)
- 与 PyTorch 集成时，可参考 `torch.compile`、自定义算子等相关文档

## 许可

如无特殊说明，示例代码可按你个人需要选择许可证；若完全私有学习可不必添加 `LICENSE` 文件。
