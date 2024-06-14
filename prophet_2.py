import pandas as pd
from prophet import Prophet

# 示例数据
df = pd.DataFrame({
    'ds': pd.date_range(start='2023-01-01', periods=30, freq='D'),
    'y': [10, 15, 20, 25, 30, 25, 20, 15, 10, 20, 25, 30, 35, 40, 35, 30, 25, 20, 15, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
})

# 假期数据
holidays = pd.DataFrame({
    'holiday': 'example_holiday',
    'ds': pd.to_datetime(['2023-01-05', '2023-01-15', '2023-01-25']),
    'lower_window': 0,
    'upper_window': 1,
})

# 初始化模型并添加假期信息
model = Prophet(holidays=holidays)

# 拟合模型
model.fit(df)

# 创建未来数据框
future = model.make_future_dataframe(periods=6, freq='H')

# 进行预测
forecast = model.predict(future)

# 可视化结果
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# 保存图像
fig1.savefig('forecast_plot-2.png')
fig2.savefig('components_plot-2.png')
