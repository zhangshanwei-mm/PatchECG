
import matplotlib.pyplot as plt
import numpy as np

# 示例数据，数据维度为[6,5]
data = np.random.rand(6, 5)

# 创建箱线图
plt.boxplot(data.T, labels=[f'Type {i+1}' for i in range(6)])

# 设置标题和标签
plt.title('Boxplot for 6 Types with 5-Fold Validation')
plt.xlabel('Type')
plt.ylabel('Values')

# 显示图形
plt.show()