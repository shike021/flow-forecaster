import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pmdarima import auto_arima
import json
import matplotlib.pyplot as plt


# 生成连续7天，每天24个点的真实客流数据
history_data = {}
start_date = datetime(2024, 4, 1)

''''''
history_data["2024-04-01"] = [ 149, 180, 212, 237, 269, 254, 306, 322, 370, 347, 382, 407, 373, 317, 321, 300, 239, 201, 205, 163, 158, 113, 108, 87 ]
history_data["2024-04-02"] = [ 139, 183, 180, 200, 271, 288, 312, 298, 336, 362, 400, 355, 330, 340, 296, 296, 262, 201, 207, 184, 147, 126, 107, 77 ] 
history_data["2024-04-03"] = [ 142, 178, 201, 208, 244, 303, 296, 310, 316, 366, 368, 413, 380, 327, 303, 260, 231, 216, 188, 172, 137, 129, 103, 75 ] 
history_data["2024-04-04"] = [ 151, 162, 194, 234, 264, 307, 287, 339, 352, 411, 407, 379, 363, 345, 273, 303, 265, 229, 180, 167, 143, 111, 109, 77 ] 
history_data["2024-04-05"] = [ 147, 171, 189, 211, 264, 281, 292, 339, 359, 367, 414, 347, 367, 348, 303, 278, 246, 208, 201, 177, 161, 119, 92, 80 ]
history_data["2024-04-06"] = [ 186, 213, 241, 292, 289, 329, 383, 370, 407, 371, 377, 290, 305, 281, 241, 231, 193, 195, 164, 124, 96, 80, 64, 37 ]
history_data["2024-04-07"] = [ 202, 229, 232, 305, 292, 290, 346, 386, 384, 347, 346, 292, 299, 253, 271, 214, 192, 193, 161, 128, 109, 75, 61, 42 ] 
history_data["2024-04-08"] = [ 147, 179, 191, 200, 226, 283, 282, 311, 344, 410, 375, 343, 334, 336, 313, 269, 239, 224, 218, 165, 142, 113, 105, 84 ]
history_data["2024-04-09"] = [ 141, 167, 216, 226, 236, 258, 294, 299, 353, 414, 364, 360, 323, 347, 286, 299, 274, 205, 201, 181, 163, 120, 109, 84 ] 
history_data["2024-04-10"] = [ 140, 178, 219, 214, 262, 285, 307, 337, 375, 411, 403, 414, 382, 338, 283, 280, 258, 207, 207, 165, 136, 128, 95, 84 ] 
history_data["2024-04-11"] = [ 154, 180, 218, 202, 250, 296, 275, 322, 319, 395, 375, 381, 347, 325, 272, 253, 248, 208, 189, 187, 157, 110, 103, 87 ] 
history_data["2024-04-12"] = [ 151, 181, 187, 207, 261, 272, 321, 350, 339, 378, 400, 408, 333, 339, 274, 275, 242, 222, 195, 167, 136, 116, 97, 76 ]
history_data["2024-04-13"] = [ 219, 236, 267, 280, 315, 316, 339, 392, 434, 391, 344, 321, 289, 297, 271, 205, 182, 185, 152, 116, 101, 81, 58, 42 ]
history_data["2024-04-14"] = [ 181, 207, 242, 294, 287, 305, 377, 385, 418, 403, 384, 295, 310, 280, 263, 239, 204, 177, 146, 121, 101, 85, 57, 41 ] 
history_data["2024-04-15"] = [ 147, 163, 192, 226, 274, 307, 301, 345, 367, 376, 368, 414, 346, 294, 296, 300, 255, 214, 215, 179, 139, 116, 96, 75 ] 
history_data["2024-04-16"] = [ 197, 215, 245, 275, 314, 337, 347, 389, 418, 411, 391, 405, 355, 321, 314, 296, 258, 230, 219, 184, 153, 131, 110, 91 ] 
history_data["2024-04-17"] = [ 150, 162, 194, 234, 264, 317, 287, 339, 352, 411, 407, 379, 363, 345, 273, 303, 265, 229, 180, 167, 143, 111, 109, 77 ] 
history_data["2024-04-18"] = [ 140, 178, 219, 214, 262, 285, 307, 337, 375, 411, 403, 414, 382, 338, 283, 280, 258, 207, 207, 165, 136, 128, 95, 84 ] 
history_data["2024-04-19"] = [ 154, 180, 218, 202, 250, 296, 275, 322, 319, 395, 370, 381, 347, 325, 272, 253, 240, 208, 189, 187, 157, 105, 103, 87 ] 
history_data["2024-04-20"] = [ 202, 229, 232, 305, 292, 290, 346, 386, 384, 347, 346, 292, 299, 253, 271, 214, 192, 193, 161, 128, 109, 75, 61, 42 ] 
history_data["2024-04-21"] = [ 189, 207, 242, 294, 287, 305, 377, 385, 418, 403, 384, 295, 310, 280, 263, 239, 204, 177, 146, 121, 101, 85, 57, 41 ] 
history_data["2024-04-22"] = [ 145, 183, 180, 200, 271, 288, 312, 290, 336, 362, 400, 355, 305, 340, 296, 296, 262, 201, 207, 181, 147, 126, 107, 70 ] 
history_data["2024-04-23"] = [ 139, 183, 180, 200, 271, 288, 312, 298, 336, 362, 400, 355, 330, 340, 296, 296, 262, 201, 207, 184, 147, 126, 107, 77 ] 
history_data["2024-04-24"] = [ 142, 178, 201, 208, 244, 303, 296, 310, 316, 366, 368, 413, 380, 327, 303, 260, 231, 216, 188, 172, 137, 129, 103, 75 ] 
history_data["2024-04-25"] = [ 151, 162, 194, 234, 264, 307, 287, 339, 352, 411, 407, 379, 363, 345, 273, 303, 265, 229, 180, 167, 143, 111, 109, 77 ] 
history_data["2024-04-26"] = [ 147, 171, 189, 211, 264, 281, 292, 339, 359, 367, 414, 347, 367, 348, 303, 278, 246, 208, 201, 177, 161, 119, 92, 80 ]
history_data["2024-04-27"] = [ 186, 213, 241, 292, 289, 329, 383, 370, 407, 371, 377, 290, 305, 281, 241, 231, 193, 195, 164, 124, 96, 80, 64, 37 ]
history_data["2024-04-28"] = [ 202, 229, 232, 305, 292, 290, 346, 386, 384, 347, 346, 292, 299, 253, 271, 214, 192, 193, 161, 128, 109, 75, 61, 42 ] 
''''''

# history_data_json = json.dumps([
#     {"date": date_str] = values} for date_str, values in history_data.items()
# ], indent=4)
# print(history_data_json)


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
# stepwise_model = auto_arima(df['value'],
#                             start_p=1, start_q=1, start_d=None,
#                             max_p=5, max_q=5, max_d=2,
#                             max_P=2, max_Q=2, max_D=1,
#                             start_P=0, start_Q=1, start_D=None,
#                             m=24,  # 每天24个点
#                             seasonal=True, seasonal_test='ocsb', # ocsb | ch
#                             stationary=True,
#                             # information_criterion='aic', # aic | bic | hqic
#                             test='kpss', # kpss | adf | pp
#                             alpha=0.05,
#                             trend='Node', # None | c | t | ct

#                             d=1, D=1, trace=True,
#                             error_action='ignore',  
#                             suppress_warnings=True, 
#                             n_jobs=2,
#                             stepwise=True)

stepwise_model = auto_arima(df['value'], start_p=1, start_q=1,
                            max_p=5, max_q=5, m=1,  # 每天24个点
                            start_P=0, seasonal=True, seasonal_test='ocsb', # ocsb | ch
                            d=0, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            test='kpss', # kpss | adf | pp
                            # trend='Node', # None | c | t | ct
                            n_jobs=2,
                            stepwise=True)

# 输出最优参数
print("Best ARIMA Parameters:", stepwise_model.order, stepwise_model.seasonal_order)
print(f"Best Model AIC: {stepwise_model.aic()}")
print(f"Best Model BIC: {stepwise_model.bic()}")


# 使用最优参数训练模型
stepwise_model_fit = stepwise_model.fit(df['value'])

# 预测未来7天（每天24个点）的客流
# forecast_steps = 7 * 24  # 7天每天24个点
forecast_steps = 120 # 越策未来7小时
forecast = stepwise_model_fit.predict(n_periods=forecast_steps)

# 将预测结果取整
forecast = np.round(forecast).astype(int)

# 将预测结果整合到与历史数据相同格式的数据结构中
last_date = df.index[-1]
forecast_dates = [last_date + timedelta(hours=i+1) for i in range(forecast_steps)]
forecast_values = list(forecast)

forecast_data = {}
for date, value in zip(forecast_dates, forecast_values):
    date_str = date.strftime('%Y-%m-%d')
    if date_str not in forecast_data:
        forecast_data[date_str] = []
    forecast_data[date_str].append(value)

# 将结果输出为JSON格式
output_data = []
for date_str, values in forecast_data.items():
    output_data.append({
        'date': date_str,
        'values': values
    })

# 输出JSON
print(json.dumps(output_data, indent=4))
