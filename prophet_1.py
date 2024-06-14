import pandas as pd
from prophet import Prophet
import plotly

# 安装 Plotly，如果尚未安装
# !pip install plotly

# 示例数据
df = pd.DataFrame({
    'ds': pd.date_range(start='2023-01-01', periods=30, freq='D'),
    'y': [10, 15, 20, 25, 30, 25, 20, 15, 10, 20, 25, 30, 35, 40, 35, 30, 25, 20, 15, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
})

# 初始化模型
model = Prophet()

# 拟合模型
model.fit(df)

# 创建未来数据框
future = model.make_future_dataframe(periods=1, freq='D')

# 进行预测
forecast = model.predict(future)

# 可视化结果
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# 保存图像
fig1.savefig('forecast_plot.png')
fig2.savefig('components_plot.png')