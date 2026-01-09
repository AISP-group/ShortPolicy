import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 40, 100)
y1 = np.sin(x) * 50 + 50
y2 = np.cos(x) * 50 + 50
y3 = np.tan(x / 10) * 20 + 60

# 计算标准差或误差范围
error = np.random.rand(len(x)) * 10

plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(x, y1, label='Series 1', color='red')
plt.plot(x, y2, label='Series 2', color='blue')
plt.plot(x, y3, label='Series 3', color='green')

# 添加置信区间（阴影区域）
plt.fill_between(x, y1 - error, y1 + error, color='red', alpha=0.2)
plt.fill_between(x, y2 - error, y2 + error, color='blue', alpha=0.2)
plt.fill_between(x, y3 - error, y3 + error, color='green', alpha=0.2)

# 设置标签和标题
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Line Plot with Confidence Intervals')
plt.legend()

# 显示图形
plt.show()
