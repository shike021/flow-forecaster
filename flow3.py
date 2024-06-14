import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pmdarima import auto_arima

# 生成连续7天，每天24个点的真实客流数据
history_data = {}
start_date = datetime(2024, 4, 1)

# 设定每天的基础客流模式，可以根据实际情况调整
base_pattern = [150, 180, 200, 220, 250, 280, 300, 320, 350, 380, 400, 380, 
                350, 320, 300, 280, 250, 220, 200, 180, 150, 120, 100, 80]

for i in range(7):  # 7天的数据
    current_date = start_date + timedelta(days=i)
    date_str = current_date.strftime('%Y-%m-%d')
    
    # 模拟每天客流随时间的波动
    day_values = []
    for hour in range(24):
        base_value = base_pattern[hour % len(base_pattern)]
        # 在基础值上加上随机波动，波动范围为基础值的正负10%
        value = base_value * (1 + 0.2 * (np.random.random() - 0.5))
        day_values.append(value)
    
    history_data[date_str] = day_values
history_data["2024-04-08"] = [149,180,197,216,238,276,305,309,336,373,400,370,369,326,302,275,243,212,189,192,151,112,98,79]
history_data["2024-04-09"] = [140,188,203,216,234,269,298,326,344,392,402,364,371,319,290,264,235,210,197,193,154,112,101,80]


# 将数据转换为DataFrame
data_list = []
for date_str, values in history_data.items():
    for hour in range(24):
        date_time = datetime.strptime(f"{date_str} {hour:02d}:00", '%Y-%m-%d %H:%M')
        data_list.append({'date': date_time, 'value': values[hour]})

df = pd.DataFrame(data_list).set_index('date').sort_index()

# 确保数据集是时间序列格式
df.index = pd.DatetimeIndex(df.index)

# 使用auto_arima函数进行模型拟合
stepwise_model = auto_arima(df['value'], start_p=1, start_q=1,
                            max_p=5, max_q=5, m=24,  # 每天24个点
                            start_P=0, seasonal=True, d=1, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)

# 输出最优参数
print("Best ARIMA Parameters:", stepwise_model.order, stepwise_model.seasonal_order)

# 使用最优参数训练模型
stepwise_model_fit = stepwise_model.fit(df['value'])

# 预测未来2天（每天24个点）的客流
forecast_steps = 2 * 24  # 2天每天24个点
forecast = stepwise_model_fit.predict(n_periods=forecast_steps)

# 输出预测结果
forecast_index = pd.date_range(df.index[-1] + timedelta(hours=1), periods=forecast_steps, freq='H')
forecast_df = pd.DataFrame({'date': forecast_index, 'forecast_value': forecast})
print(forecast_df)
