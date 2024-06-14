import pandas as pd
from prophet import Prophet

# 假设我们有客流量和温度数据
df = pd.DataFrame({
    'ds': pd.date_range(start='2023-01-01', periods=365, freq='D'),
    'y': [30 + i % 10 for i in range(365)],  # 假设的客流量数据，这里简化为随机生成的数据
    'temperature': [5 + 10 * (i % 7) for i in range(365)]  # 假设的温度数据，每周变化一次
})

# 初始化 Prophet 模型
model = Prophet()

# 添加额外的回归因子（温度）
model.add_regressor('temperature')

# 拟合模型
model.fit(df)

# 创建未来数据框（预测未来7天）
future = model.make_future_dataframe(periods=7)

# 手动生成未来的温度数据，确保长度正确
future['temperature'] = [15 + 10 * ((len(df) + i) % 7) for i in range(len(future))]

# 进行预测
forecast = model.predict(future)

# 打印预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'temperature', 'trend']])

# 可视化结果
fig = model.plot(forecast)


# 保存图像
fig.savefig('forecast_plot_4.png')
