# Learn_ViT

基于 PyTorch 的 Vision Transformer (ViT) 迁移学习实战项目，用于**植物叶片病害分类**。项目从 timm 的 ViT 源码出发，配合自定义数据集加载、训练/验证流程、指标可视化与推理脚本，完整覆盖了「数据 → 训练 → 评估 → 预测」的端到端流程。

## 项目特点

- **完整的 ViT 实现**：`model/vit_model.py` 来自 [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)，包含 PatchEmbed、Multi-Head Attention、MLP、Transformer Block 等组件，并附详细中文注释。
- **迁移学习友好**：自动加载 ImageNet-21k 预训练权重，支持冻结 backbone 仅微调分类头；并对「权重/模型不匹配」做了**智能检测**（从权重推断 `patch_size / embed_dim / depth`，给出 `--model` 修正建议）。
- **完善的训练流程**：SGD + 余弦退火学习率，每轮记录 `train/val` 的 loss、acc、precision、recall、F1 到 `metrics.csv`，并保存 `last.pth` / `best.pth` checkpoint。
- **自动可视化**：训练结束后自动绘制 loss 曲线、acc 曲线、PRF 曲线，以及基于 best 权重的混淆矩阵。
- **稳健的推理脚本**：默认严格校验权重与模型结构是否匹配，防止选错模型版本；支持单张图片或文件夹批量预测，可把预测类别与置信度写回图片。

## 目录结构

```
Learn_ViT
├── model/
│   └── vit_model.py            # ViT 模型定义（PatchEmbed / Block / 工厂函数等）
├── tools/
│   ├── my_dataset.py           # 自定义 Dataset 与 DataLoader 构建
│   ├── utils.py                # 数据划分、train_one_epoch、evaluate、ConsolePrinter
│   ├── create_exp_folder.py    # 实验目录管理（run/train/exp, exp1, exp2 ...）
│   └── plot_metrics.py         # 曲线 / 混淆矩阵可视化
├── weights/
│   └── jx_vit_base_patch16_224_in21k-e5005f0a.pth   # 预训练权重
├── Plant_Leaf_Disease/         # 数据集（按文件夹组织，每个子目录一个类别）
├── class_counts.csv            # 各类别样本数统计
├── train.py                    # 训练入口
├── predict.py                  # 推理入口
├── requirements.txt
└── run/                        # 训练输出（自动生成）
    └── train/exp/
        ├── metrics.csv
        ├── loss_curve.png / acc_curve.png / val_prf_curve.png
        ├── confusion_matrix.png
        ├── class_indices.json
        └── weights/{last.pth, best.pth}
```

## 数据集

`Plant_Leaf_Disease` 共 **25 个类别**，涵盖甜椒、茄子、柑橘、玉米、生菜、土豆、番茄等作物的健康/病害叶片，例如：

- `Bell_pepper__Bell_pepper_Bacterial_spot`（甜椒-细菌性斑点病）
- `Corn__Corn_Common_rust`（玉米-锈病）
- `Tomato__Tomato_Late_blight`（番茄-晚疫病）
- `Lettuce__fungal`（生菜-真菌感染）
- ……

数据按 `类别文件夹/图片` 的标准 ImageFolder 结构组织，`tools/utils.py:read_split_data` 会按 `val_rate=0.2` 划分训练/验证集，并写入 `class_indices.json`。

## 环境依赖

```bash
pip install -r requirements.txt
```

PyTorch 安装请按显卡型号选择（见 `requirements.txt` 注释）：

| 显卡世代           | torch 版本 | CUDA |
| ------------------ | ---------- | ---- |
| 30 / 40 系列       | 1.10.1     | 11.3 |
| 40 系列            | 2.0.1      | 11.8 |
| 50 系列（及更新）  | 2.7.1      | 12.8 |

> Python 推荐 3.10。国内可用镜像源加速：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

## 快速开始

### 1. 训练

```bash
python train.py \
    --epochs 20 \
    --batch-size 256 \
    --lr 0.001 \
    --data-path Plant_Leaf_Disease \
    --model vit_base_patch16_224_in21k \
    --weights weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth \
    --freeze-layers True \
    --device cuda:0
```

主要参数：

| 参数              | 默认值                                              | 说明                                              |
| ----------------- | --------------------------------------------------- | ------------------------------------------------- |
| `--epochs`        | 20                                                  | 训练轮数                                          |
| `--batch-size`    | 256                                                 | 批大小（显存不足请调小）                          |
| `--lr` / `--lrf`  | 0.001 / 0.01                                        | 初始学习率 / cosine 末端比例                      |
| `--data-path`     | `Plant_Leaf_Disease`                                | 数据集根目录                                      |
| `--model`         | `vit_base_patch16_224_in21k`                        | ViT 工厂函数名（B/16、B/32、L/16、L/32、H/14 等） |
| `--weights`       | `weights/jx_vit_base_patch16_224_in21k-...pth`      | 预训练权重路径，传空字符串则从头训练              |
| `--freeze-layers` | True                                                | 冻结 backbone，仅训练 head / pre_logits           |
| `--device`        | `cuda:0`                                            | 设备，无 GPU 自动回退到 cpu                       |

可选模型工厂函数：`vit_base_patch16_224_in21k`、`vit_base_patch32_224_in21k`、`vit_large_patch16_224_in21k`、`vit_large_patch32_224_in21k`、`vit_huge_patch14_224_in21k`。

训练产物会写入 `run/train/exp/`（已存在则递增为 `exp1`、`exp2` …）。

### 2. 预测

```bash
python predict.py \
    --data path/to/image_or_folder \
    --weights run/train/exp/weights/best.pth \
    --class-indices run/train/exp/class_indices.json \
    --model-name vit_base_patch16_224_in21k \
    --device cuda:0 \
    --draw
```

- `--data` 支持单张图片或文件夹（递归遍历）。
- `--model-name` **必须与训练时一致**，否则严格校验会报错并列出 shape mismatch 详情。
- `--draw` 会把预测类别与置信度绘制到图片上并保存。
- 类别数优先从权重 `head.weight` 推断；无法推断时回退到 `class_indices.json` 或 `--num-classes`。
- 结果输出到 `run/val/exp/`（递增命名）：`predictions.txt`（`image_path  pred_id  pred_name  prob`）及标注后的图片。

## 训练指标与可视化

训练过程中每个 epoch 会向 `metrics.csv` 写入：

```
epoch, train_loss, train_acc, val_loss, val_acc, val_p, val_r, val_f1, lr
```

训练结束后自动生成：

- `loss_curve.png` / `acc_curve.png`：训练/验证 loss 与 acc 曲线（含平滑曲线）
- `val_prf_curve.png`：验证集 Precision / Recall / F1 曲线
- `confusion_matrix.png`：基于 best 权重在验证集上的混淆矩阵

## 权重加载的安全机制

`train.py` 中的 `_smart_load_weights` 与 `predict.py` 中的 `safe_load_state_dict` 都做了多层防护：

1. 兼容纯 `state_dict` 与含 `model_state` 的 checkpoint，自动剥离 `module.` 前缀（DataParallel 兼容）。
2. 自动剔除分类头 `head.weight / head.bias`（类别数必然不匹配）。
3. 按 shape 过滤后计算 `keep_ratio`，低于 85% 直接报错，并从权重推断结构特征给出 `--model` 建议。
4. `predict.py` 默认严格匹配（任何 missing / unexpected / mismatched 都报错），需要部分加载可加 `--allow-partial-load`。

## 致谢

- ViT 原始实现：[rwightman/pytorch-image-models (timm)](https://github.com/rwightman/pytorch-image-models)
- 论文：*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., 2020)
