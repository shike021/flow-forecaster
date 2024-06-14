import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# 生成真实的历史数据
def generate_realistic_history_data(start_date, num_days):
    history_data = {}
    date_range = pd.date_range(start=start_date, periods=num_days, freq='D')
    
    for date in date_range:
        if date.weekday() < 5:  # 工作日
            daily_data = [np.random.randint(100, 300) for _ in range(6)] + \
                         [np.random.randint(300, 500) for _ in range(10)] + \
                         [np.random.randint(100, 300) for _ in range(8)]
        else:  # 周末
            daily_data = [np.random.randint(150, 350) for _ in range(6)] + \
                         [np.random.randint(350, 550) for _ in range(10)] + \
                         [np.random.randint(150, 350) for _ in range(8)]
        
        history_data[date.strftime('%Y-%m-%d')] = daily_data
    
    return history_data

# 将数据转换为DataFrame
def convert_to_dataframe(history_data):
    data_list = []
    for date_str, values in history_data.items():
        for idx, value in enumerate(values):
            hour = idx
            data_list.append({'date': datetime.strptime(date_str, '%Y-%m-%d') + timedelta(hours=hour), 'value': value})
    
    df = pd.DataFrame(data_list).set_index('date').sort_index()
    return df

# 生成14天的历史数据
start_date = '2024-04-01'
num_days = 14
history_data = generate_realistic_history_data(start_date, num_days)
df = convert_to_dataframe(history_data)

# 训练 ARIMA 模型
arima_order = (5, 1, 0)
model = ARIMA(df['value'], order=arima_order)
model_fit = model.fit()

# 预测未来7天
forecast_steps = 24 * 7  # 未来7天，每天24小时
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=forecast_steps, freq='h')
forecast_series = pd.Series(forecast, index=forecast_index)

# 绘制历史数据和预测结果
plt.figure(figsize=(16, 8))
plt.plot(df.index, df['value'], label='Historical Data')
plt.plot(forecast_series.index, forecast_series, label='Predicted Data', color='red')
plt.fill_betweenx([0, max(df['value'])], start_date, (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=7)).strftime('%Y-%m-%d'), color='grey', alpha=0.2)
plt.title('Passenger Flow Forecast')
plt.xlabel('Date')
plt.ylabel('Passenger Count')
plt.legend()
plt.show()
