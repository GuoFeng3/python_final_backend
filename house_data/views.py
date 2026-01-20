from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .models import HouseDeal
import pandas as pd
import json
import numpy as np
from datetime import date
from django.db.models import Count, Avg, F, FloatField, ExpressionWrapper, Case, When, Value, CharField
from django.db.models.functions import TruncMonth, TruncQuarter
from dateutil.relativedelta import relativedelta

from django.views.decorators.csrf import csrf_exempt
import csv
import io
from dateutil import parser as date_parser


@csrf_exempt
def add_data(request):
    """
    接收CSV文件并存入数据库
    参数：file (CSV文件)
    """
    if request.method != 'POST':
        return JsonResponse({'error': '仅支持POST请求'}, status=405, json_dumps_params={'ensure_ascii': False})

    file = request.FILES.get('file')
    if not file:
        return JsonResponse({'error': '未上传文件'}, status=400, json_dumps_params={'ensure_ascii': False})

    city_param = request.POST.get('city')

    try:
        rows = []
        if file.name.endswith(('.xlsx', '.xls')):
            try:
                # 读取Excel文件
                # dtype=str 确保所有字段读取为字符串，fillna('') 处理空值
                df = pd.read_excel(file, dtype=str)
                df = df.fillna('')
                rows = df.to_dict('records')
            except ImportError:
                return JsonResponse({'error': '缺少 openpyxl 库，无法读取 Excel 文件'}, status=500,
                                    json_dumps_params={'ensure_ascii': False})
            except Exception as e:
                return JsonResponse({'error': f'读取Excel文件失败: {str(e)}'}, status=400,
                                    json_dumps_params={'ensure_ascii': False})
        else:
            # 读取CSV文件
            decoded_file = file.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)
            rows = csv.DictReader(io_string)

        house_deals = []
        for row in rows:
            # 数据清洗与转换
            try:
                # 辅助函数：安全获取字符串值
                def get_val(key):
                    val = row.get(key, '')
                    return str(val).strip() if val is not None else ''

                # 处理面积：去除 '㎡'
                area_str = get_val('area').replace('㎡', '')
                area = float(area_str) if area_str else 0.0

                # 处理成交价：去除 '万'
                price_str = get_val('deal_price').replace('万', '')
                deal_price = int(float(price_str)) if price_str else 0

                # 处理日期
                deal_date_str = get_val('deal_date')
                deal_date = None
                if deal_date_str and deal_date_str.lower() != 'nan':
                    try:
                        # 尝试使用 dateutil 解析日期
                        deal_date = date_parser.parse(deal_date_str).date()
                    except (ValueError, TypeError):
                        # 如果解析失败，打印错误并跳过日期字段
                        print(f"Invalid date format: {deal_date_str}, row: {row}")
                        continue

                # 确定城市
                # 优先级：参数 > CSV列 > 默认(北京)
                city_val = city_param if city_param else get_val('city')
                if not city_val:
                    city_val = '北京'

                # 创建模型实例
                deal = HouseDeal(
                    house_id=get_val('house_id'),
                    title=get_val('title'),
                    city=city_val,
                    district=get_val('district'),
                    district_area=get_val('district_area'),
                    layout=get_val('layout'),
                    area=area,
                    floor=get_val('floor'),
                    direction=get_val('direction'),
                    deal_price=deal_price,
                    deal_unit_price=get_val('deal_unit_price'),
                    deal_date=deal_date
                )
                house_deals.append(deal)
            except ValueError as e:
                print(f"Skipping row due to error: {e}, row: {row}")
                continue

        # 批量插入数据库
        if house_deals:
            HouseDeal.objects.bulk_create(house_deals)

        return JsonResponse({'message': f'成功导入 {len(house_deals)} 条数据'},
                            json_dumps_params={'ensure_ascii': False})

    except Exception as e:
        return JsonResponse({'error': f'处理文件时出错: {str(e)}'}, status=500,
                            json_dumps_params={'ensure_ascii': False})

def prediction_comparison_view(request):
    """
    返回预测对比的前端页面
    """
    return render(request, 'house_data/prediction_comparison.html')
def total_any(request):
    """
    返回分析后的整体统计：平均单价、单价中位数、平均总价、总价中位数，平均面积
    参数：city (可选，默认所有或北京，根据需求。这里如果没传则统计所有，如果传了则统计该城市)
    参数：district (行政区名，或 'all')
    """
    city = request.GET.get('city')
    district = request.GET.get('district') or request.GET.get('行政区名')

    # 1. 基础查询构造
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district and district.lower() != 'all':
        query = query.filter(district=district)

    # 2. 获取数据 (仅获取需要的字段，使用 values_list 减少开销)
    # 返回元组列表 [(price, area), ...]
    data = list(query.values_list('deal_price', 'area'))

    if not data:
        return JsonResponse({
            'avg_unit_price': 0,
            'median_unit_price': 0,
            'avg_deal_price': 0,
            'median_deal_price': 0,
            'avg_area': 0,
            'deal_count': 0
        }, json_dumps_params={'ensure_ascii': False})

    # 3. 使用 Numpy 进行计算 (比 Pandas 更快且内存占用更小)
    # data 是 list of tuples
    # 转为 numpy array
    # shape: (N, 2)
    arr = np.array(data, dtype=np.float64)

    deal_prices = arr[:, 0]
    areas = arr[:, 1]

    # 计算单价
    unit_prices = deal_prices * 10000.0 / areas

    # 计算统计指标
    stats = {
        'avg_unit_price': round(float(np.mean(unit_prices)), 2),
        'median_unit_price': round(float(np.median(unit_prices)), 2),
        'avg_deal_price': round(float(np.mean(deal_prices)), 2),
        'median_deal_price': round(float(np.median(deal_prices)), 2),
        'avg_area': round(float(np.mean(areas)), 2),
        'deal_count': len(data)
    }

    return JsonResponse(stats, json_dumps_params={'ensure_ascii': False})


def total_avg_price(request):
    """
    返回各城区平均单价排名（从高到低）
    参数：city (可选)
    """
    city = request.GET.get('city')

    # 1. 基础查询构造
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    # 2. 数据库聚合计算
    # 按 district 分组，计算平均单价
    district_stats = query.values('district').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        )
    ).order_by('-avg_unit_price')

    # 3. 结果格式化
    result = []
    for item in district_stats:
        if item['district']:
            price = round(item['avg_unit_price'], 2) if item['avg_unit_price'] is not None else 0
            result.append({
                'district': item['district'],
                'unit_price': price
            })

    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})


def predict_price_linear(request):
    """
    使用简单线性回归算法进行近六个月房价预测
    参数：district (行政区名，或 'all')
    参数：city (可选)
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')

    if not district_name:
        district_name = 'all'

    # 1. 获取历史月度数据
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)

    # 按月统计平均单价
    trend_data = query.annotate(
        month=TruncMonth('deal_date')
    ).values('month').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        )
    ).order_by('month')

    # 转换为列表并清洗数据
    data_list = []
    for item in trend_data:
        if item['month'] and item['avg_unit_price'] is not None:
            data_list.append({
                'month': item['month'],
                'price': float(item['avg_unit_price'])
            })

    if len(data_list) < 2:
        return JsonResponse({'error': '数据不足，无法进行预测'}, status=400, json_dumps_params={'ensure_ascii': False})

    # 2. 准备线性回归数据
    # 使用时间序号作为 X (0, 1, 2, ...)
    X = np.arange(len(data_list))
    Y = np.array([d['price'] for d in data_list])

    # 3. 线性拟合 (1次多项式即线性回归)
    # z = [slope, intercept]
    try:
        z = np.polyfit(X, Y, 1)
        p = np.poly1d(z)
    except Exception as e:
        return JsonResponse({'error': f'线性回归计算失败: {str(e)}'}, status=500,
                            json_dumps_params={'ensure_ascii': False})

    # 4. 预测未来6个月
    predictions = []
    last_date = data_list[-1]['month']
    last_x = X[-1]

    for i in range(1, 7):
        next_x = last_x + i
        pred_price = p(next_x)

        # 价格不应为负
        if pred_price < 0:
            pred_price = 0

        next_date = last_date + relativedelta(months=i)

        predictions.append({
            'month': next_date.strftime('%Y-%m'),
            'price': round(pred_price, 2)
        })

    # 5. 构建返回数据 (包含历史数据和预测数据)
    # 历史数据从 2023-01 开始
    history_start_date = date(2023, 1, 1)
    history = []
    for item in data_list:
        if item['month'] >= history_start_date:
            history.append({
                'month': item['month'].strftime('%Y-%m'),
                'price': round(item['price'], 2)
            })

    return JsonResponse({
        'history': history,
        'prediction': predictions
    }, safe=False, json_dumps_params={'ensure_ascii': False})


def predict_price_lstm(request):
    """
    使用 Keras LSTM 算法进行近六个月房价预测
    LSTM (Long Short-Term Memory) 是一种特殊的 RNN，擅长捕捉长期依赖关系，适合时间序列预测。

    参数：district (行政区名，或 'all')
    参数：city (可选)
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')

    if not district_name:
        district_name = 'all'

    # 1. 获取历史月度数据
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)

    trend_data = query.annotate(
        month=TruncMonth('deal_date')
    ).values('month').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        )
    ).order_by('month')

    data_list = []
    for item in trend_data:
        if item['month'] and item['avg_unit_price'] is not None:
            data_list.append({
                'month': item['month'],
                'price': float(item['avg_unit_price'])
            })

    # 需要足够的数据来构建序列 (Window size 3 + target 1 = 4 points min for 1 sample)
    if len(data_list) < 5:
        return JsonResponse({'error': '数据不足(至少需要5个月)，无法进行LSTM预测'}, status=400,
                            json_dumps_params={'ensure_ascii': False})

    # 2. 引入 TensorFlow (延迟引入)
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        import os
        # 禁用 GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    except ImportError as e:
        return JsonResponse({'error': f'TensorFlow 未安装或加载失败: {str(e)}'}, status=500,
                            json_dumps_params={'ensure_ascii': False})

    # 3. 数据预处理
    prices = np.array([d['price'] for d in data_list])
    min_val = np.min(prices)
    max_val = np.max(prices)

    # 归一化 [0, 1]
    if max_val - min_val == 0:
        norm_prices = np.zeros_like(prices)
    else:
        norm_prices = (prices - min_val) / (max_val - min_val)

    # 构建滑动窗口数据集
    window_size = 6
    X_train = []
    y_train = []

    for i in range(len(norm_prices) - window_size):
        X_train.append(norm_prices[i:i + window_size])
        y_train.append(norm_prices[i + window_size])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Reshape for LSTM: (samples, time_steps, features)
    X_train = X_train.reshape((X_train.shape[0], window_size, 1))

    # 4. 构建模型
    # 设置随机种子
    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential([
        LSTM(units=50, activation='tanh', input_shape=(window_size, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # 5. 训练模型
    # LSTM 参数较多，适当增加 epoch
    model.fit(X_train, y_train, epochs=200, batch_size=4, verbose=0)

    # 6. 预测未来 6 个月
    predictions = []
    last_date = data_list[-1]['month']

    # 初始输入窗口
    current_window = norm_prices[-window_size:]

    for i in range(1, 7):
        # 预测下一步
        input_seq = current_window.reshape((1, window_size, 1))
        pred_norm = model.predict(input_seq, verbose=0)[0][0]

        # 反归一化
        pred_price = pred_norm * (max_val - min_val) + min_val
        if pred_price < 0:
            pred_price = 0

        next_date = last_date + relativedelta(months=i)

        predictions.append({
            'month': next_date.strftime('%Y-%m'),
            'price': round(float(pred_price), 2)
        })

        # 更新窗口
        current_window = np.append(current_window[1:], pred_norm)

    # 7. 构建返回数据 (包含历史数据和预测数据)
    # 历史数据从 2023-01 开始
    history_start_date = date(2023, 1, 1)
    history = []
    for item in data_list:
        if item['month'] >= history_start_date:
            history.append({
                'month': item['month'].strftime('%Y-%m'),
                'price': round(item['price'], 2)
            })

    return JsonResponse({
        'history': history,
        'prediction': predictions
    }, safe=False, json_dumps_params={'ensure_ascii': False})


def direction_price(request):
    """
    获取指定行政区（或所有）的各朝向平均单价
    参数：district (行政区名，或 'all')
    参数：city (可选)
    返回：各朝向的平均单价
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')

    if not district_name:
        return JsonResponse({'error': '缺少参数：行政区名'}, status=400, json_dumps_params={'ensure_ascii': False})

    # 1. 基础查询构造
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)

    # 2. 数据库聚合计算
    # 按 direction 分组
    # 注意：数据库中的 direction 可能包含空格，如果数据很脏可能需要Trim，但通常'南'就是'南'
    direction_stats = query.values('direction').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        )
    ).order_by('-avg_unit_price')

    # 3. 结果格式化
    result = []
    for item in direction_stats:
        if item['direction']:
            # Python端简单清洗：去除空格
            direction = item['direction'].strip()
            price = round(item['avg_unit_price'], 2) if item['avg_unit_price'] is not None else 0
            result.append({
                'direction': direction,
                'avg_unit_price': price
            })

    merged_data = {}
    for item in result:
        d = item['direction']
        p = item['avg_unit_price']
        if d not in merged_data:
            merged_data[d] = {'total': 0, 'count': 0}
        merged_data[d]['total'] += p
        merged_data[d]['count'] += 1

    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})


def quarter_trend(request):
    """
    获取指定行政区（或所有）的季度平均单价趋势
    参数：district (行政区名，或 'all')
    参数：city (可选)
    返回：从最早交易季度到最晚交易季度的季度平均单价
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')

    if not district_name:
        district_name = 'all'

    # 1. 基础查询构造
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)

    # 2. 数据库聚合计算
    # 使用 TruncQuarter 将日期截断为季度
    trend_data = query.annotate(
        quarter=TruncQuarter('deal_date')
    ).values('quarter').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        )
    ).order_by('quarter')

    # 3. 结果格式化
    result = []
    for item in trend_data:
        if item['quarter']:
            # TruncQuarter 返回的是该季度的第一天 (date类型)
            # 我们需要将其转换为 "YYYYQx" 格式
            d = item['quarter']
            quarter_str = f"{d.year}Q{(d.month - 1) // 3 + 1}"

            price = round(item['avg_unit_price'], 2) if item['avg_unit_price'] is not None else 0
            result.append({
                'quarter': quarter_str,
                'avg_unit_price': price
            })

    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})


def month_trend(request):
    """
    获取指定行政区的月度平均单价趋势
    参数：district (行政区名，或 'all')
    参数：city (可选)
    返回：从最早交易月份到最晚交易月份的月平均单价
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')

    if not district_name:
        district_name = 'all'

    # 1. 基础查询构造
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)
    trend_data = query.annotate(
        month=TruncMonth('deal_date')
    ).values('month').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        )
    ).order_by('month')
    result = []
    for item in trend_data:
        if item['month']:
            month_str = item['month'].strftime('%Y-%m')
            # 处理可能的 None 值
            price = round(item['avg_unit_price'], 2) if item['avg_unit_price'] is not None else 0
            result.append({
                'month': month_str,
                'avg_unit_price': price
            })

    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})


def district_area_rank(request):
    """
    获取指定行政区下各细分区域的平均单价排名
    参数：district (行政区名，或 'all')
    参数：city (可选)
    返回：细分区域名（去除行政区前缀）和平均单价
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')

    if not district_name:
        district_name = 'all'

    # 1. 基础查询构造
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)

    # 2. 数据库聚合计算
    # 按 district_area 分组，同时获取平均单价和数量（用于后续加权合并）
    area_stats = query.values('district_area').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        ),
        count=Count('id')
    )

    # 3. 结果格式化与Python端后处理（合并同名区域）
    merged_stats = {}

    for item in area_stats:
        raw_name = item['district_area']
        price = item['avg_unit_price']
        count = item['count']

        if not raw_name or price is None or count == 0:
            continue

        # 清洗：去除 '-' 前缀
        clean_name = raw_name.split('-')[-1] if '-' in str(raw_name) else raw_name

        if clean_name not in merged_stats:
            merged_stats[clean_name] = {'weighted_sum': 0.0, 'total_count': 0}

        merged_stats[clean_name]['weighted_sum'] += price * count
        merged_stats[clean_name]['total_count'] += count

    # 4. 生成最终结果列表
    result = []
    for name, data in merged_stats.items():
        if data['total_count'] > 0:
            avg_price = data['weighted_sum'] / data['total_count']
            result.append({
                'district_area': name,
                'avg_unit_price': round(avg_price, 2)
            })

    # 5. 排序
    result.sort(key=lambda x: x['avg_unit_price'], reverse=True)

    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})


def predict_price(request):
    """
    根据行政区历史月均价预测未来三个月房价
    使用 Numpy 进行线性回归预测
    参数：district (行政区名，或 'all')
    参数：city (可选)
    参数：layout (户型，可选，默认为'all')
    返回：未来六个月的预测均价
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')
    layout = request.GET.get('layout')

    if not district_name:
        district_name = 'all'

    if not layout:
        layout = 'all'

    # 1. 基础查询构造
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)

    if layout.lower() != 'all':
        query = query.filter(layout=layout)

    # 2. 数据库聚合计算：按月平均单价
    # 使用 TruncMonth
    monthly_stats = query.annotate(
        month=TruncMonth('deal_date')
    ).values('month').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        )
    ).order_by('month')

    # 转换为列表供 numpy 使用
    # 过滤掉 month 为 None 的 (如果有)
    stats_list = []
    for item in monthly_stats:
        if item['month'] and item['avg_unit_price'] is not None:
            stats_list.append({
                'month': item['month'],
                'unit_price': item['avg_unit_price']
            })

    if len(stats_list) < 2:
        return JsonResponse({'error': '数据不足，无法预测'}, status=400, json_dumps_params={'ensure_ascii': False})

    # 3. 准备特征预测模型数据
    # 原始线性回归误差较大，升级为"趋势+季节性"特征预测模型
    # 模型: Price = w0 + w1*t + w2*t^2 + w3*sin(m) + w4*cos(m)
    n_samples = len(stats_list)
    x_t = np.arange(n_samples)
    y = np.array([item['unit_price'] for item in stats_list])

    predicted_data = []

    # 只有当数据量足够时(>=6个点)才使用复杂模型，否则降级为二次多项式或线性回归
    if n_samples >= 6:
        # 提取月份特征 (1-12)
        months = np.array([item['month'].month for item in stats_list])

        # 特征工程
        X_bias = np.ones(n_samples)
        X_t = x_t
        X_t2 = x_t ** 2
        X_sin = np.sin(2 * np.pi * months / 12)
        X_cos = np.cos(2 * np.pi * months / 12)

        # 构造特征矩阵 (N, 5)
        X = np.column_stack([X_bias, X_t, X_t2, X_sin, X_cos])

        # 求解最小二乘
        # rcond=None 让 numpy 自动处理奇异值
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # 预测未来 6 个月
        last_month_date = stats_list[-1]['month']

        for i in range(1, 7):
            next_date = last_month_date + relativedelta(months=i)
            t_next = n_samples - 1 + i
            m_next = next_date.month

            # 构造预测向量
            feat = np.array([
                1,
                t_next,
                t_next ** 2,
                np.sin(2 * np.pi * m_next / 12),
                np.cos(2 * np.pi * m_next / 12)
            ])

            pred_price = np.dot(feat, w)

            # 防止价格预测为负数 (极端情况)
            if pred_price < 0:
                pred_price = 0

            predicted_data.append({
                'month': next_date.strftime('%Y-%m'),
                'price': round(pred_price, 2)
            })

    else:
        # 数据较少时，使用二次多项式拟合 (Quadratic Regression)
        # 比单纯线性回归好，能捕捉一定弯曲趋势
        deg = 2 if n_samples >= 3 else 1
        z = np.polyfit(x_t, y, deg)
        p = np.poly1d(z)

        last_month_date = stats_list[-1]['month']
        for i in range(1, 7):  # 6个月
            next_date = last_month_date + relativedelta(months=i)
            t_next = n_samples - 1 + i
            pred_price = p(t_next)
            # 防止价格预测为负数
            if pred_price < 0:
                pred_price = 0

            predicted_data.append({
                'month': next_date.strftime('%Y-%m'),
                'price': round(pred_price, 2)
            })

    # 构建返回数据 (包含历史数据和预测数据)
    # 历史数据从 2023-01 开始
    history_start_date = date(2023, 1, 1)
    history = []
    for item in stats_list:
        if item['month'] >= history_start_date:
            history.append({
                'month': item['month'].strftime('%Y-%m'),
                'price': round(item['unit_price'], 2)
            })

    return JsonResponse({
        'history': history,
        'prediction': predicted_data
    }, safe=False, json_dumps_params={'ensure_ascii': False})


def squaremeter_avgprice(request):
    """
    按面积区间统计平均单价、平均总价和成交量
    参数：district (行政区名，或 'all')
    参数：city (可选)
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')

    if not district_name:
        district_name = 'all'

    # 1. 基础查询构造
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)

    # 2. 数据库聚合计算
    # 使用 Case/When 进行分组
    stats = query.annotate(
        area_range=Case(
            When(area__lt=50, then=Value('50㎡以下')),
            When(area__lt=70, then=Value('50-70㎡')),
            When(area__lt=90, then=Value('70-90㎡')),
            When(area__lt=110, then=Value('90-110㎡')),
            When(area__lt=150, then=Value('110-150㎡')),
            default=Value('150㎡以上'),
            output_field=CharField(),
        )
    ).values('area_range').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        ),
        avg_total_price=Avg('deal_price'),
        count=Count('id')
    )

    # 3. 结果格式化与排序
    # 定义排序顺序
    order_map = {
        '50㎡以下': 1,
        '50-70㎡': 2,
        '70-90㎡': 3,
        '90-110㎡': 4,
        '110-150㎡': 5,
        '150㎡以上': 6
    }

    result = []
    for item in stats:
        price_unit = round(item['avg_unit_price'], 2) if item['avg_unit_price'] is not None else 0
        price_total = round(item['avg_total_price'], 2) if item['avg_total_price'] is not None else 0

        result.append({
            'area_range': item['area_range'],
            'avg_unit_price': price_unit,
            'avg_total_price': price_total,
            'count': item['count']
        })

    # Python 端排序
    result.sort(key=lambda x: order_map.get(x['area_range'], 99))

    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})


def room_hall_any(request):
    """
    返回对应区划不同房型的成交量和平均单价，按照成交量排序
    参数：district (行政区名，或 'all')
    参数：city (可选)
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')

    if not district_name:
        district_name = 'all'

    # 1. 基础查询构造
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)

    # 过滤掉 layout 为空的数据
    # 过滤掉字符串形式的 'nan', 'NaN', 'null' 等 (数据库中可能是字符串)
    query = query.exclude(layout__isnull=True).exclude(layout__in=['', 'nan', 'NaN', 'null', 'None'])

    # 2. 数据库聚合计算
    layout_stats = query.values('layout').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        ),
        count=Count('id')
    ).order_by('-count')

    # 3. 结果格式化
    result = []
    for item in layout_stats:
        # 二次检查 layout 是否真的有效（虽然 exclude 已经过滤了大部分）
        layout_val = item['layout']
        if not layout_val or str(layout_val).lower() in ['nan', 'none', 'null', '']:
            continue

        price = round(item['avg_unit_price'], 2) if item['avg_unit_price'] is not None else 0
        result.append({
            'layout': layout_val,
            'avg_unit_price': price,
            'count': item['count']
        })

    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})


def predict_price_rnn(request):
    """
    使用 Keras SimpleRNN 算法进行近六个月房价预测
    参数：district (行政区名，或 'all')
    参数：city (可选)
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')

    if not district_name:
        district_name = 'all'

    # 1. 获取历史月度数据
    query = HouseDeal.objects.filter(area__gt=0)
    if city:
        query = query.filter(city=city)

    if district_name.lower() != 'all':
        query = query.filter(district=district_name)

    trend_data = query.annotate(
        month=TruncMonth('deal_date')
    ).values('month').annotate(
        avg_unit_price=Avg(
            ExpressionWrapper(
                F('deal_price') * 10000.0 / F('area'),
                output_field=FloatField()
            )
        )
    ).order_by('month')

    data_list = []
    for item in trend_data:
        if item['month'] and item['avg_unit_price'] is not None:
            data_list.append({
                'month': item['month'],
                'price': float(item['avg_unit_price'])
            })

    # 需要足够的数据来构建序列 (Window size 3 + target 1 = 4 points min for 1 sample)
    # 建议至少 5-6 个月数据
    if len(data_list) < 5:
        return JsonResponse({'error': '数据不足(至少需要5个月)，无法进行RNN预测'}, status=400,
                            json_dumps_params={'ensure_ascii': False})

    # 2. 引入 TensorFlow (延迟引入，避免影响启动速度)
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import SimpleRNN, Dense
        import os
        # 禁用 GPU (对于简单任务，CPU 更快且无驱动烦恼)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    except ImportError as e:
        return JsonResponse({'error': f'TensorFlow 未安装或加载失败: {str(e)}'}, status=500,
                            json_dumps_params={'ensure_ascii': False})

    # 3. 数据预处理
    prices = np.array([d['price'] for d in data_list])
    min_val = np.min(prices)
    max_val = np.max(prices)

    # 归一化 [0, 1]
    if max_val - min_val == 0:
        norm_prices = np.zeros_like(prices)
    else:
        norm_prices = (prices - min_val) / (max_val - min_val)

    # 构建滑动窗口数据集
    window_size = 6
    X_train = []
    y_train = []

    for i in range(len(norm_prices) - window_size):
        X_train.append(norm_prices[i:i + window_size])
        y_train.append(norm_prices[i + window_size])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Reshape for RNN: (samples, time_steps, features)
    X_train = X_train.reshape((X_train.shape[0], window_size, 1))

    # 4. 构建模型
    # 设置随机种子
    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential([
        SimpleRNN(units=50, activation='tanh', input_shape=(window_size, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # 5. 训练模型
    # 数据量少，因此选择50轮
    model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)

    # 6. 预测未来 6 个月
    predictions = []
    last_date = data_list[-1]['month']

    # 初始输入窗口
    current_window = norm_prices[-window_size:]

    for i in range(1, 7):
        # 预测下一步
        # reshape (1, 3, 1)
        input_seq = current_window.reshape((1, window_size, 1))
        pred_norm = model.predict(input_seq, verbose=0)[0][0]

        # 反归一化
        pred_price = pred_norm * (max_val - min_val) + min_val
        if pred_price < 0:
            pred_price = 0

        next_date = last_date + relativedelta(months=i)

        predictions.append({
            'month': next_date.strftime('%Y-%m'),
            'price': round(float(pred_price), 2)
        })

        # 更新窗口: 移除第一个，加入新预测值
        current_window = np.append(current_window[1:], pred_norm)

    # 7. 构建返回数据 (包含历史数据和预测数据)
    # 历史数据从 2023-01 开始
    history_start_date = date(2023, 1, 1)
    history = []
    for item in data_list:
        if item['month'] >= history_start_date:
            history.append({
                'month': item['month'].strftime('%Y-%m'),
                'price': round(item['price'], 2)
            })

    return JsonResponse({
        'history': history,
        'prediction': predictions
    }, safe=False, json_dumps_params={'ensure_ascii': False})
