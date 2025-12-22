from django.urls import path
from . import views

urlpatterns = [
    # 总数据统计（平均单价、单价中位数、平均总价、总价中位数，平均面积）
    path('total-any/', views.total_any, name='total_any'),
    # 各城区平均单价排名（如东城：10000）
    path('total-avg-price/', views.total_avg_price, name='total_avg_price'),
    # 各城区月交易趋势（如：2023年1月，2023年2月）
    path('month-trend/', views.month_trend, name='month_trend'),
    #各行政区的各小区划（如昌平-长阳，昌平-回龙观）房价排名
    path('district-area-rank/', views.district_area_rank, name='district_area_rank'),
    # 线性回归模型预测未来三个月房价
    path('predict-price/', views.predict_price, name='predict_price'),
    # 各城区按平米区间计算平米单价 如：50-70平米，70-100平米，100-200平米
    path('squaremeter-avgprice/', views.squaremeter_avgprice, name='squaremeter_avgprice'),
    # 各城区按季度（如：2023Q1，2023年Q2）计算交易趋势
    path('quarter-trend/', views.quarter_trend, name='quarter_trend'),
    # 各城区按房屋朝向（南、北）方向计算交易价格
    path('direction-price/', views.direction_price, name='direction_price'),
    # 添加数据，csv或xlsx文件
    path('add-data/', views.add_data, name='add_data'),
]
