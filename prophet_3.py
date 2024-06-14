import pandas as pd
from prophet import Prophet

# 生成300个小时的客流历史数据
df = pd.DataFrame({
    'ds': pd.date_range(start='2023-01-01', periods=300, freq='H'),
    'y': [i % 24 + (i // 24) * 5 + (i % 24 > 20) * 10 for i in range(300)]
})

# 模拟假期数据
holidays = pd.DataFrame({
    'holiday': 'example_holiday',
    'ds': pd.to_datetime(['2023-01-05 00:00:00', '2023-01-15 00:00:00', '2023-01-25 00:00:00']),
    'lower_window': 0,
    'upper_window': 1,
})

# 添加周末
weekends = pd.DataFrame({
    'holiday': 'weekend',
    'ds': pd.to_datetime([
        '2023-01-07 00:00:00', '2023-01-08 00:00:00', 
        '2023-01-14 00:00:00', '2023-01-15 00:00:00', 
        '2023-01-21 00:00:00', '2023-01-22 00:00:00', 
        '2023-01-28 00:00:00', '2023-01-29 00:00:00'
    ]),
    'lower_window': 0,
    'upper_window': 1,
})

# 合并假期和周末
holidays = pd.concat([holidays, weekends])

# 初始化模型并添加假期和周末信息  
# 置信区间默认0.80
#model = Prophet(holidays=holidays) 
# 初始化模型并设置置信区间为 95%
model = Prophet(holidays=holidays, interval_width=0.95)

# 拟合模型
model.fit(df)

# 创建未来数据框
future = model.make_future_dataframe(periods=10, freq='H') #未来10小时

# 进行预测
forecast = model.predict(future)

# 可视化结果
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)


# 保存图像
fig1.savefig('forecast_plot_3-2.png')
fig2.savefig('components_plot_3-2.png')

# 获取特定时间点的预测信息
# 例如，获取第一个未来预测点的信息
specific_time = future['ds'].iloc[-10]  # 获取第一个未来预测点的时间戳
specific_forecast = forecast[forecast['ds'] == specific_time]

print(specific_forecast)