import os
import sys
import json
import random

import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

# 控制台打印参数类
class ConsolePrinter:
    """
    用于 train/val 控制台打印的格式化工具（表头/数值对齐 + 颜色）
    """

    def __init__(self,
                 sep="  ",
                 w_epoch=7, w_loss=7, w_acc=7, w_size=4, w_prf=7,
                 c_train="\033[96m", c_val="\033[93m",
                 bold="\033[1m", reset="\033[0m"):
        """初始化打印格式参数（列宽、分隔符、颜色等）。

        Args:
            sep: 列与列之间的分隔字符串
            w_epoch/w_loss/w_acc/w_size/w_prf: 各列宽度
            c_train/c_val: train/val 行的 ANSI 颜色码
            bold/reset: 加粗 / 重置 ANSI 码

        Returns:
            None
        """
        self.SEP = sep
        self.W_EPOCH = w_epoch
        self.W_LOSS = w_loss
        self.W_ACC = w_acc
        self.W_SIZE = w_size
        self.W_PRF = w_prf

        self.C_TRAIN = c_train
        self.C_VAL = c_val
        self.BOLD = bold
        self.RESET = reset
        self.BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    def color(self, text, c) -> str:
        """给文本套上颜色（加粗 + 颜色码 + 重置）。

        Args:
            text: 原始字符串
            c: ANSI 颜色码

        Returns:
            str: 带颜色转义码的字符串
        """
        return f"{self.BOLD}{c}{text}{self.RESET}"

    # ---------- Train ----------
    def train_header(self, colored=True) -> str:
        """生成训练表头字符串（epoch / loss / Acc / Size）。

        Args:
            colored: 是否着色

        Returns:
            str: 训练表头
        """
        s = (
            f"{'epoch':<{self.W_EPOCH}}{self.SEP}"
            f"{'loss':>{self.W_LOSS}}{self.SEP}"
            f"{'Acc':>{self.W_ACC}}{self.SEP}"
            f"{'Size':>{self.W_SIZE}}"
        )
        return self.color(s, self.C_TRAIN) if colored else s

    def train_desc(self, epoch_idx, epochs, loss, acc, size) -> str:
        """生成训练进度条描述行（当前 epoch/loss/acc/size 数值）。

        Args:
            epoch_idx: 当前 epoch（从 1 开始）
            epochs: 总 epoch 数
            loss: 当前累计平均 loss
            acc: 当前累计平均准确率
            size: 输入图像尺寸

        Returns:
            str: 训练描述行
        """
        ep = f"{epoch_idx}/{epochs}"
        return (
            f"{ep:<{self.W_EPOCH}}{self.SEP}"
            f"{loss:>{self.W_LOSS}.3f}{self.SEP}"
            f"{acc:>{self.W_ACC}.3f}{self.SEP}"
            f"{size:>{self.W_SIZE}d}"
        )

    # ---------- Val ----------
    def val_header(self, colored=True, keep_size_placeholder=True) -> str:
        """生成验证表头字符串（loss / Acc / P / R / F1）。

        Args:
            colored: 是否着色
            keep_size_placeholder: 是否保留 size 列占位，使 P/R/F1 与 train 对齐

        Returns:
            str: 验证表头
        """
        # keep_size_placeholder=True：保留 size 占位，让 P/R/F1 与 train 的 size 列对齐
        if keep_size_placeholder:
            s = (
                f"{'':<{self.W_EPOCH}}{self.SEP}"
                f"{'loss':>{self.W_LOSS}}{self.SEP}"
                f"{'Acc':>{self.W_ACC}}{self.SEP}"
                f"{'':>{self.W_SIZE}}{self.SEP}"
                f"{'P':>{self.W_PRF}}{self.SEP}"
                f"{'R':>{self.W_PRF}}{self.SEP}"
                f"{'F1':>{self.W_PRF}}"
            )
        else:
            # 更紧凑：不留 size 占位，P/R/F1 更靠近 loss/acc（你之前问“为什么离得远”就是这个）
            s = (
                f"{'':<{self.W_EPOCH}}{self.SEP}"
                f"{'loss':>{self.W_LOSS}}{self.SEP}"
                f"{'Acc':>{self.W_ACC}}{self.SEP}"
                f"{'P':>{self.W_PRF}}{self.SEP}"
                f"{'R':>{self.W_PRF}}{self.SEP}"
                f"{'F1':>{self.W_PRF}}"
            )

        return self.color(s, self.C_VAL) if colored else s

    def val_desc(self, loss, acc, p, r, f1, keep_size_placeholder=True) -> str:
        """生成验证进度条描述行（loss/acc/P/R/F1 数值）。

        Args:
            loss: 当前累计平均 loss
            acc: 当前累计平均准确率
            p/r/f1: 宏平均 precision / recall / f1
            keep_size_placeholder: 是否保留 size 列占位

        Returns:
            str: 验证描述行
        """
        if keep_size_placeholder:
            return (
                f"{'':<{self.W_EPOCH}}{self.SEP}"
                f"{loss:>{self.W_LOSS}.3f}{self.SEP}"
                f"{acc:>{self.W_ACC}.3f}{self.SEP}"
                f"{'':>{self.W_SIZE}}{self.SEP}"
                f"{p:>{self.W_PRF}.3f}{self.SEP}"
                f"{r:>{self.W_PRF}.3f}{self.SEP}"
                f"{f1:>{self.W_PRF}.3f}"
            )
        else:
            return (
                f"{'':<{self.W_EPOCH}}{self.SEP}"
                f"{loss:>{self.W_LOSS}.3f}{self.SEP}"
                f"{acc:>{self.W_ACC}.3f}{self.SEP}"
                f"{p:>{self.W_PRF}.3f}{self.SEP}"
                f"{r:>{self.W_PRF}.3f}{self.SEP}"
                f"{f1:>{self.W_PRF}.3f}"
            )


def read_split_data(
    root: str,
    val_rate: float = 0.2,
    exp_folder: Optional[str] = None,
    seed: int = 0,
    allowed_exts: Tuple[str, ...] = (".jpg", ".JPG", ".png", ".PNG")
    ) -> Tuple[List[str], List[int], List[str], List[int], int]:
    """
    扫描 ImageFolder 风格数据集并划分 train/val（按类别分层抽样）。

    数据集目录结构示例：
      root/
        class_a/ xxx.jpg ...
        class_b/ yyy.jpg ...

    Args:
        root: 数据集根目录
        val_rate: 验证集比例（每个类别内部按比例抽）
        exp_folder: 如果提供，则把 class_indices.json 写到 exp_folder 下
        seed: 随机种子，保证划分可复现
        allowed_exts: 允许的图片后缀

    Returns:
        train_images_path, train_images_label, val_images_path, val_images_label
    """
    assert os.path.exists(root), f"dataset root: {root} does not exist."
    assert 0.0 <= val_rate < 1.0, "val_rate should be in [0, 1)."

    # 使用局部随机数生成器，避免影响全局 random 状态，也更利于复现
    rng = random.Random(seed)

    # 找到所有类别文件夹（每个文件夹一个类别）
    class_names = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    # 排序保证跨平台顺序一致
    class_names.sort()

    # 构建 class_name -> class_id 的映射
    class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    num_classes = len(class_names)

    # 保存 idx -> class_name 到 json（方便可视化/推理时反查类别名）
    # 写到 exp_folder下，如果 exp_folder=None 则默认写到root下
    save_dir = exp_folder if exp_folder is not None else root
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "class_indices.json")
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=4, ensure_ascii=False)

    # 准备输出容器
    train_images_path: List[str] = []
    train_images_label: List[int] = []
    val_images_path: List[str] = []
    val_images_label: List[int] = []
    class_counts: List[int] = []  # 每类样本数统计

    # 遍历每个类别，按类别内部比例抽 val（分层划分）
    for class_name in class_names:
        class_dir = os.path.join(root, class_name)

        # 收集该类别下所有支持后缀的图片路径
        images = [
            os.path.join(class_dir, fn)
            for fn in os.listdir(class_dir)
            if os.path.splitext(fn)[-1] in allowed_exts
        ]
        images.sort()  # 排序保证一致性

        class_id = class_to_idx[class_name]
        n = len(images)
        class_counts.append(n)

        if n == 0:
            # 某个类别文件夹里没有图片，直接跳过
            continue

        # 计算该类别要抽多少张做 val
        # 目标：尽量按比例抽；但要避免出现“某类 val=0”或“某类 train=0”
        if val_rate == 0.0:
            k = 0
        else:
            # 至少抽 1 张val（n>=2 时），同时至少保留 1 张 train
            # -n=1：只能放到train（否则train会空）
            if n >= 2:
                k = int(n * val_rate)
                k = max(1, k)          # 至少 1 张 val
                k = min(k, n - 1)      # 至少留 1 张给 train
            else:
                k = 0

        # 随机抽样得到验证集图片（用 set 加速 membership 判断）
        val_samples = set(rng.sample(images, k=k)) if k > 0 else set()

        # 分配到 train / val
        for img_path in images:
            if img_path in val_samples:
                val_images_path.append(img_path)
                val_images_label.append(class_id)
            else:
                train_images_path.append(img_path)
                train_images_label.append(class_id)

    # 打印统计信息
    total = sum(class_counts)
    print(f"{total} images were found in the dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")
    print(f"class_indices.json saved to: {json_path}")

    # 基本合法性检查
    assert len(train_images_path) > 0, "number of training images must be greater than 0."
    assert len(val_images_path) > 0, "number of validation images must be greater than 0. " \
                                     "Try reducing val_rate or check dataset."

    return train_images_path, train_images_label, val_images_path, val_images_label, num_classes



def train_one_epoch(model, optimizer, data_loader, device, epoch, epochs) -> Tuple[float, float]:
    """在训练集上训练一个 epoch。

    Args:
        model: 待训练的模型
        optimizer: 优化器
        data_loader: 训练集 DataLoader
        device: 训练设备
        epoch: 当前 epoch 索引（从 0 开始）
        epochs: 总 epoch 数

    Returns:
        Tuple[float, float]: (该 epoch 的平均 loss, 平均准确率)
    """
    printer = ConsolePrinter()
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1, device=device)
    accu_num  = torch.zeros(1, device=device)
    sample_num = 0

    optimizer.zero_grad()

    pbar = tqdm(data_loader, file=sys.stdout, dynamic_ncols=True,
                bar_format=printer.BAR_FORMAT, leave=True)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        sample_num += images.size(0)
        img_size = images.shape[-1]  # 假设输入方图，比如 224

        pred = model(images)
        pred_classes = pred.argmax(dim=1)
        accu_num += (pred_classes == labels).sum()

        loss = loss_function(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accu_loss += loss.detach()

        # 动态显示当前“累计均值”
        avg_loss = accu_loss.item() / (step + 1)
        avg_acc  = accu_num.item() / sample_num

        desc = printer.train_desc(epoch + 1, epochs, avg_loss, avg_acc, img_size)
        pbar.set_description_str(desc)

        if not torch.isfinite(loss):
            print(f"\nWARNING: non-finite loss, ending training: {loss}")
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



def _macro_prf_from_cm(cm: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从混淆矩阵计算宏平均 precision / recall / f1。

    Args:
        cm: 混淆矩阵 [K, K]，行为真实、列为预测
        eps: 防除零的小常数

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (macro_precision, macro_recall, macro_f1)
        只对验证集中出现过的类（support>0）做宏平均
    """
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    support = cm.sum(1).float()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    mask = support > 0  # 只对 val 中出现过的类做宏平均
    if mask.any():
        return precision[mask].mean(), recall[mask].mean(), f1[mask].mean()
    else:
        z = cm.new_tensor(0.0).float()
        return z, z, z


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, epochs, num_classes: int, indent_spaces: int = 16) -> Tuple[float, float, float, float, float]:
    """在验证集上评估模型，返回 loss/acc 与宏平均 P/R/F1。

    Args:
        model: 待评估模型
        data_loader: 验证集 DataLoader
        device: 评估设备
        epoch: 当前 epoch 索引
        epochs: 总 epoch 数
        num_classes: 类别数 K
        indent_spaces: 进度条描述缩进空格数（保留参数）

    Returns:
        Tuple[float, float, float, float, float]:
        (avg_loss, avg_acc, macro_precision, macro_recall, macro_f1)
    """
    printer = ConsolePrinter()
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1, device=device)
    accu_num  = torch.zeros(1, device=device)
    sample_num = 0

    cm = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    pbar = tqdm(data_loader, file=sys.stdout, dynamic_ncols=True,
                bar_format=printer.BAR_FORMAT, leave=True)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        sample_num += images.size(0)

        pred = model(images)
        pred_classes = pred.argmax(dim=1)

        accu_num += (pred_classes == labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        # 更新混淆矩阵（高效）
        idx = labels * num_classes + pred_classes
        cm += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        avg_loss = accu_loss.item() / (step + 1)
        avg_acc  = accu_num.item() / sample_num
        mp, mr, mf = _macro_prf_from_cm(cm)

        desc = printer.val_desc(avg_loss, avg_acc, mp, mr, mf, keep_size_placeholder=True)
        pbar.set_description_str(desc)

    avg_loss = accu_loss.item() / (step + 1)
    avg_acc  = accu_num.item() / sample_num
    mp, mr, mf = _macro_prf_from_cm(cm)

    return avg_loss, avg_acc, float(mp.item()), float(mr.item()), float(mf.item())