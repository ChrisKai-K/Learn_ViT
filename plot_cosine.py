import math
import matplotlib.pyplot as plt
import numpy as np


def lf(x, epochs=50, lrf=0.01):
    return ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf


# 生成数据
epochs = 50
lrf = 0.01
x = np.linspace(0, epochs, 500)
y = [lf(xi, epochs, lrf) for xi in x]

# 绘图
plt.figure(figsize=(10, 4))
plt.plot(x, y, 'b-', linewidth=2, label=f'lrf={lrf}')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate Factor')
plt.title('Cosine Annealing Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/xingyubo/Documents/learn/Learn_ViT/cosine_annealing.png', dpi=100)
print(f"图像已保存到: /Users/xingyubo/Documents/learn/Learn_ViT/cosine_annealing.png")