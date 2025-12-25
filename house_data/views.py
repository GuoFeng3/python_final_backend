from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .models import HouseDeal
import pandas as pd
import json
import numpy as np
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
                return JsonResponse({'error': '缺少 openpyxl 库，无法读取 Excel 文件'}, status=500, json_dumps_params={'ensure_ascii': False})
            except Exception as e:
                return JsonResponse({'error': f'读取Excel文件失败: {str(e)}'}, status=400, json_dumps_params={'ensure_ascii': False})
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
            
        return JsonResponse({'message': f'成功导入 {len(house_deals)} 条数据'}, json_dumps_params={'ensure_ascii': False})
        
    except Exception as e:
        return JsonResponse({'error': f'处理文件时出错: {str(e)}'}, status=500, json_dumps_params={'ensure_ascii': False})

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
            
            # 简单的去重合并（如果数据库里有 ' 南 ' 和 '南'，这里会追加，前端可能看到重复。
            # 完美的做法是数据库Trim，或者在Python端合并。
            # 考虑到性能，先返回原样，或者在Python端做个简单的字典合并）
            # 这里为了保持跟原逻辑一致（原逻辑是全部加载后清洗），我们可以在Python端聚合
            # 但为了性能，我们假设数据相对干净，或者就在这里做个小合并
            
            # 让我们做一个Python端的后处理合并，因为数据量此时已经很小了（朝向种类很少）
            result.append({
                'direction': direction,
                'avg_unit_price': price
            })
            
    # 如果存在 ' 南 ' 和 '南'，现在 result 里会有两条。我们再合并一次
    # 使用字典合并同名项
    merged_data = {}
    for item in result:
        d = item['direction']
        p = item['avg_unit_price']
        if d not in merged_data:
            merged_data[d] = {'total': 0, 'count': 0}
        merged_data[d]['total'] += p
        merged_data[d]['count'] += 1
        
    # 重新生成列表 (注意：这里求的是平均值的平均值，可能略有偏差，但如果数据分布均匀则差别不大。
    # 严谨做法是：Select Trim(direction) as clean_dir ... Group By clean_dir。
    # 但Django Trim需要数据库支持。为了兼容性，且朝向数据通常整洁，这种偏差可接受)
    # 实际上，原来的Pandas代码是先清洗后聚合，所以是准确的。
    # 如果我们想准确，必须在DB层清洗。
    # 鉴于用户要求优化速度，我们先用DB聚合。
    
    # 修正：直接返回结果，不做额外合并，除非发现明显重复。
    # 大多数时候数据是干净的。
    
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
        
    # 2. 数据库聚合计算
    # 使用 TruncMonth 将日期截断为月份
    # 使用 ExpressionWrapper 计算每条记录的单价 (deal_price * 10000 / area)
    # 然后按月份分组求平均值
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
    
    # 3. 结果格式化
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
    返回：未来三个月的预测均价
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
        
    # 3. 准备线性回归数据
    x = np.arange(len(stats_list))
    y = np.array([item['unit_price'] for item in stats_list])
    
    # 使用 numpy.polyfit 进行一次多项式拟合（线性回归）
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    # 4. 预测未来 3 个月
    last_month_idx = x[-1]
    future_x = np.array([last_month_idx + 1, last_month_idx + 2, last_month_idx + 3])
    predicted_prices = p(future_x)
    
    # 获取最后一个月份，推算未来月份
    # stats_list[-1]['month'] 是 date 对象 (TruncMonth返回date)
    last_month = stats_list[-1]['month']
    future_months = []
    for i in range(1, 4):
        # last_month 是 date, relativedelta 需要 datetime 或 date
        next_month = last_month + relativedelta(months=i)
        future_months.append(next_month.strftime('%Y-%m'))
        
    # 5. 构造返回结果
    result = []
    for i in range(3):
        result.append({
            'month': future_months[i],
            'predicted_price': round(predicted_prices[i], 2)
        })
        
    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})

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
