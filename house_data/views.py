from django.shortcuts import render
from django.http import JsonResponse
from .models import HouseDeal
import pandas as pd
import numpy as np
from django.db.models import F
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
    
    # 获取所有房屋数据的价格和面积
    query = HouseDeal.objects.all()
    if city:
        query = query.filter(city=city)
        
    if district and district.lower() != 'all':
        query = query.filter(district=district)
        
    data = query.values('deal_price', 'area')
    
    # 转换为DataFrame
    df = pd.DataFrame(list(data))
    
    if df.empty:
        return JsonResponse({
            'avg_unit_price': 0,
            'median_unit_price': 0,
            'avg_deal_price': 0,
            'median_deal_price': 0,
            'avg_area': 0
        }, json_dumps_params={'ensure_ascii': False})

    # 数据清洗：确保面积大于0
    df = df[df['area'] > 0]
    
    # 计算单价 (deal_price是万元，area是平米，单价通常是元/平米)
    df['unit_price'] = df['deal_price'] * 10000 / df['area']
    
    # 计算统计指标
    stats = {
        'avg_unit_price': round(df['unit_price'].mean(), 2),
        'median_unit_price': round(df['unit_price'].median(), 2),
        'avg_deal_price': round(df['deal_price'].mean(), 2),
        'median_deal_price': round(df['deal_price'].median(), 2),
        'avg_area': round(df['area'].mean(), 2)
    }
    
    return JsonResponse(stats, json_dumps_params={'ensure_ascii': False})

def total_avg_price(request):
    """
    返回各城区平均单价排名（从高到低）
    参数：city (可选)
    """
    city = request.GET.get('city')
    
    # 获取所有房屋数据的城区、价格和面积
    query = HouseDeal.objects.all()
    if city:
        query = query.filter(city=city)
        
    data = query.values('district', 'deal_price', 'area')
    
    # 转换为DataFrame
    df = pd.DataFrame(list(data))
    
    if df.empty:
        return JsonResponse([], safe=False, json_dumps_params={'ensure_ascii': False})

    # 数据清洗：确保面积大于0
    df = df[df['area'] > 0]
    
    # 计算单价
    df['unit_price'] = df['deal_price'] * 10000 / df['area']
    
    # 按城区进行分组并计算平均单价
    district_stats = df.groupby('district')['unit_price'].mean().reset_index()
    
    
    # 保留两位小数
    district_stats['unit_price'] = district_stats['unit_price'].round(2)
    
    # 按平均单价从高到低排序
    district_stats = district_stats.sort_values(by='unit_price', ascending=False)
    
    # 转换为字典列表
    result = district_stats.to_dict(orient='records')
    
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
        
    # 根据参数获取数据
    query = HouseDeal.objects.all()
    if city:
        query = query.filter(city=city)
        
    if district_name.lower() != 'all':
        query = query.filter(district=district_name)
        
    data = query.values('direction', 'deal_price', 'area')
        
    df = pd.DataFrame(list(data))
    
    if df.empty:
        return JsonResponse([], safe=False, json_dumps_params={'ensure_ascii': False})
        
    # 数据清洗
    df = df[df['area'] > 0]
    
    # 计算单价
    df['unit_price'] = df['deal_price'] * 10000 / df['area']
    
    # 简单的朝向清洗：去除前后空格
    df['direction'] = df['direction'].str.strip()
    
    # 按朝向分组计算平均单价
    direction_stats = df.groupby('direction')['unit_price'].mean().reset_index()
    
    # 保留两位小数
    direction_stats['unit_price'] = direction_stats['unit_price'].round(2)
    
    # 按平均单价从高到低排序
    direction_stats = direction_stats.sort_values(by='unit_price', ascending=False)
    
    # 重命名列
    direction_stats.columns = ['direction', 'avg_unit_price']
    
    # 转换为字典列表
    result = direction_stats.to_dict(orient='records')
    
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
        
    # 根据参数获取数据
    query = HouseDeal.objects.all()
    if city:
        query = query.filter(city=city)
        
    if district_name.lower() != 'all':
        query = query.filter(district=district_name)
        
    data = query.values('deal_date', 'deal_price', 'area')
        
    df = pd.DataFrame(list(data))
    
    if df.empty:
        return JsonResponse([], safe=False, json_dumps_params={'ensure_ascii': False})
        
    # 数据清洗
    df = df[df['area'] > 0]
    
    # 计算单价
    df['unit_price'] = df['deal_price'] * 10000 / df['area']
    
    # 转换日期为 datetime 类型
    df['deal_date'] = pd.to_datetime(df['deal_date'])
    
    # 提取季度 (YYYY-Qx)
    # to_period('Q') returns something like 2023Q1
    df['quarter'] = df['deal_date'].dt.to_period('Q')
    
    # 按季度分组计算平均单价
    quarterly_stats = df.groupby('quarter')['unit_price'].mean().reset_index()
    
    # 按季度排序
    quarterly_stats = quarterly_stats.sort_values(by='quarter')
    
    # 格式化季度为字符串
    quarterly_stats['quarter'] = quarterly_stats['quarter'].astype(str)
    
    # 保留两位小数
    quarterly_stats['unit_price'] = quarterly_stats['unit_price'].round(2)
    
    # 重命名列
    quarterly_stats.columns = ['quarter', 'avg_unit_price']
    
    # 转换为字典列表
    result = quarterly_stats.to_dict(orient='records')
    
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
    
    query = HouseDeal.objects.all()
    if city:
        query = query.filter(city=city)
        
    if district_name.lower() != 'all':
        query = query.filter(district=district_name)
        
    # 获取该行政区的所有交易数据
    data = query.values('deal_date', 'deal_price', 'area')
    
    df = pd.DataFrame(list(data))
    
    if df.empty:
        return JsonResponse([], safe=False, json_dumps_params={'ensure_ascii': False})
        
    # 数据清洗
    df = df[df['area'] > 0]
    
    # 计算单价
    df['unit_price'] = df['deal_price'] * 10000 / df['area']
    
    # 转换日期为 datetime 类型
    df['deal_date'] = pd.to_datetime(df['deal_date'])
    
    # 提取月份 (YYYY-MM)
    df['month'] = df['deal_date'].dt.to_period('M').astype(str)
    
    # 按月份分组计算平均单价
    monthly_stats = df.groupby('month')['unit_price'].mean().reset_index()
    
    # 保留两位小数
    monthly_stats['unit_price'] = monthly_stats['unit_price'].round(2)
    
    # 按月份排序
    monthly_stats = monthly_stats.sort_values(by='month')
    
    # 重命名列以符合通常的API习惯
    monthly_stats.columns = ['month', 'avg_unit_price']
    
    # 转换为字典列表
    result = monthly_stats.to_dict(orient='records')
    
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
        
    # 获取该行政区的所有交易数据
    query = HouseDeal.objects.all()
    if city:
        query = query.filter(city=city)
        
    if district_name.lower() != 'all':
        query = query.filter(district=district_name)
        
    data = query.values('district_area', 'deal_price', 'area')
    
    df = pd.DataFrame(list(data))
    
    if df.empty:
        return JsonResponse([], safe=False, json_dumps_params={'ensure_ascii': False})
        
    # 数据清洗
    df = df[df['area'] > 0]
    
    # 计算单价
    df['unit_price'] = df['deal_price'] * 10000 / df['area']
    
    # 处理 district_area 字段，去除 '-' 及前面的内容
    # 假设格式为 "房山-长阳"，则取 "长阳"
    df['clean_district_area'] = df['district_area'].apply(lambda x: x.split('-')[-1] if '-' in str(x) else x)
    
    # 按细分区域分组计算平均单价
    area_stats = df.groupby('clean_district_area')['unit_price'].mean().reset_index()
    
    # 保留两位小数
    area_stats['unit_price'] = area_stats['unit_price'].round(2)
    
    # 按平均单价从高到低排序
    area_stats = area_stats.sort_values(by='unit_price', ascending=False)
    
    # 重命名列
    area_stats.columns = ['district_area', 'avg_unit_price']
    
    # 转换为字典列表
    result = area_stats.to_dict(orient='records')
    
    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})

def predict_price(request):
    """
    根据行政区历史月均价预测未来三个月房价
    使用 Numpy 进行线性回归预测
    参数：district (行政区名，或 'all')
    参数：city (可选)
    返回：未来三个月的预测均价
    """
    district_name = request.GET.get('district') or request.GET.get('行政区名')
    city = request.GET.get('city')
    
    if not district_name:
        district_name = 'all'
        
    # 获取该行政区的所有交易数据
    query = HouseDeal.objects.all()
    if city:
        query = query.filter(city=city)
        
    if district_name.lower() != 'all':
        query = query.filter(district=district_name)
        
    data = query.values('deal_date', 'deal_price', 'area')
    
    df = pd.DataFrame(list(data))
    
    if df.empty or len(df) < 2:
        return JsonResponse({'error': '数据不足，无法预测'}, status=400, json_dumps_params={'ensure_ascii': False})
        
    # 数据清洗
    df = df[df['area'] > 0]
    
    # 计算单价
    df['unit_price'] = df['deal_price'] * 10000 / df['area']
    
    # 转换日期为 datetime 类型
    df['deal_date'] = pd.to_datetime(df['deal_date'])
    
    # 提取月份 (YYYY-MM)
    df['month'] = df['deal_date'].dt.to_period('M')
    
    # 按月份分组计算平均单价
    monthly_stats = df.groupby('month')['unit_price'].mean().reset_index()
    
    # 按月份排序
    monthly_stats = monthly_stats.sort_values(by='month')
    
    # 准备线性回归数据
    # X 为时间序列索引 (0, 1, 2, ...)，y 为价格
    if len(monthly_stats) < 2:
        return JsonResponse({'error': '月度数据不足，无法预测'}, status=400, json_dumps_params={'ensure_ascii': False})
        
    x = np.arange(len(monthly_stats))
    y = monthly_stats['unit_price'].values
    
    # 使用 numpy.polyfit 进行一次多项式拟合（线性回归）
    # z 是系数 [斜率, 截距]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    # 预测未来 3 个月
    last_month_idx = x[-1]
    future_x = np.array([last_month_idx + 1, last_month_idx + 2, last_month_idx + 3])
    predicted_prices = p(future_x)
    
    # 获取最后一个月份，推算未来月份
    last_month = monthly_stats['month'].iloc[-1].to_timestamp()
    future_months = []
    for i in range(1, 4):
        next_month = last_month + relativedelta(months=i)
        future_months.append(next_month.strftime('%Y-%m'))
        
    # 构造返回结果
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
        
    # 根据参数获取数据
    query = HouseDeal.objects.all()
    if city:
        query = query.filter(city=city)
        
    if district_name.lower() != 'all':
        query = query.filter(district=district_name)
        
    data = query.values('deal_price', 'area')
        
    df = pd.DataFrame(list(data))
    
    if df.empty:
        return JsonResponse([], safe=False, json_dumps_params={'ensure_ascii': False})
        
    # 数据清洗
    df = df[df['area'] > 0]
    
    # 计算单价
    df['unit_price'] = df['deal_price'] * 10000 / df['area']
    
    # 定义面积区间
    bins = [0, 50, 70, 90, 110, 150, float('inf')]
    labels = ['50㎡以下', '50-70㎡', '70-90㎡', '90-110㎡', '110-150㎡', '150㎡以上']
    
    # 对面积进行分组
    df['area_range'] = pd.cut(df['area'], bins=bins, labels=labels, right=False)
    # 注意：pandas cut默认左闭右开[a, b)，如果需要50㎡以下包含50，需要调整。
    # 通常 "50㎡以下" 是 <50, "50-70" 是 >=50 and <70.
    # right=False means [0, 50), [50, 70)...
    # 但是最后一个 150以上 应该是 >= 150. float('inf') 会覆盖所有大于150的.
    
    # 分组统计
    stats = df.groupby('area_range', observed=False).agg({
        'unit_price': 'mean',
        'deal_price': 'mean',
        'area': 'count'  # 使用任意非空列计算数量
    }).reset_index()
    
    # 重命名列
    stats.columns = ['area_range', 'avg_unit_price', 'avg_total_price', 'count']
    
    # 格式化数据
    stats['avg_unit_price'] = stats['avg_unit_price'].round(2)
    stats['avg_total_price'] = stats['avg_total_price'].round(2)
    
    # 转换为字典列表
    result = stats.to_dict(orient='records')
    
    # 过滤掉没有数据的区间（可选，根据需求，如果不希望返回NaN的区间可以过滤）
    # 这里我们把NaN的转为0或者过滤掉
    # 如果某区间没有数据，agg结果可能是NaN
    result = [
        {
            'area_range': row['area_range'],
            'avg_unit_price': row['avg_unit_price'] if not pd.isna(row['avg_unit_price']) else 0,
            'avg_total_price': row['avg_total_price'] if not pd.isna(row['avg_total_price']) else 0,
            'count': row['count']
        }
        for row in result
    ]
    
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
        
    query = HouseDeal.objects.all()
    if city:
        query = query.filter(city=city)
        
    if district_name.lower() != 'all':
        query = query.filter(district=district_name)
        
    data = query.values('layout', 'deal_price', 'area')
    
    df = pd.DataFrame(list(data))
    
    if df.empty:
        return JsonResponse([], safe=False, json_dumps_params={'ensure_ascii': False})
        
    # 数据清洗
    df = df[df['area'] > 0]
    
    # 计算单价
    df['unit_price'] = df['deal_price'] * 10000 / df['area']
    
    # 分组统计
    stats = df.groupby('layout').agg({
        'unit_price': 'mean',
        'area': 'count'
    }).reset_index()
    
    # 重命名列
    stats.columns = ['layout', 'avg_unit_price', 'count']
    
    # 排序：按成交量降序
    stats = stats.sort_values(by='count', ascending=False)
    
    # 格式化
    stats['avg_unit_price'] = stats['avg_unit_price'].round(2)
    
    # 转换为字典列表
    result = stats.to_dict(orient='records')
    
    return JsonResponse(result, safe=False, json_dumps_params={'ensure_ascii': False})
