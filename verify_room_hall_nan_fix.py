import os
import django
import pandas as pd
import json
from django.test import RequestFactory
import datetime

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from house_data.models import HouseDeal
from house_data.views import room_hall_any

def test_room_hall_nan_fix():
    print("Testing room_hall_any nan filtering...")

    # 清理旧数据 (为了测试准确性，暂时清理，或者只用特定的 house_id 区分)
    # 这里我们创建一些特定的测试数据，并在测试后清理
    test_ids = ['TEST_NAN_1', 'TEST_NAN_2', 'TEST_VALID_1']
    HouseDeal.objects.filter(house_id__in=test_ids).delete()

    # 创建测试数据
    # 1. 正常的布局
    HouseDeal.objects.create(
        house_id='TEST_VALID_1',
        title='Valid Layout House',
        city='TestCity',
        district='TestDistrict',
        district_area='TestArea',
        layout='2室1厅',
        area=100.0,
        floor='10',
        direction='South',
        deal_price=500, # 500万
        deal_date=datetime.date(2023, 1, 1)
    )

    # 2. layout 为 'nan' 字符串
    HouseDeal.objects.create(
        house_id='TEST_NAN_1',
        title='Nan String Layout House',
        city='TestCity',
        district='TestDistrict',
        district_area='TestArea',
        layout='nan',
        area=100.0,
        floor='10',
        direction='South',
        deal_price=500,
        deal_date=datetime.date(2023, 1, 1)
    )
    
    # 3. layout 为 None (如果数据库允许，或者模拟某些情况)
    # Django CharField default not null, so passing None might raise error or be saved as 'None' string if coerced?
    # Let's try to create one with empty string which is also often a problem, or just 'NaN'
    HouseDeal.objects.create(
        house_id='TEST_NAN_2',
        title='NaN String Layout House',
        city='TestCity',
        district='TestDistrict',
        district_area='TestArea',
        layout='NaN', # Case sensitivity check
        area=100.0,
        floor='10',
        direction='South',
        deal_price=500,
        deal_date=datetime.date(2023, 1, 1)
    )

    # 构造请求
    factory = RequestFactory()
    request = factory.get('/room-hall-any/', {'city': 'TestCity', 'district': 'TestDistrict'})

    # 调用视图
    response = room_hall_any(request)
    
    # 解析响应
    content = json.loads(response.content.decode('utf-8'))
    print(f"Response data: {content}")

    # 验证
    found_nan = False
    found_valid = False
    
    for item in content:
        layout = item['layout']
        print(f"Checking layout: {layout}")
        if str(layout).lower() == 'nan':
            found_nan = True
        if layout == '2室1厅':
            found_valid = True

    if found_nan:
        print("FAILED: Found 'nan' layout in response.")
    else:
        print("PASSED: No 'nan' layout found in response.")

    if found_valid:
        print("PASSED: Found valid layout in response.")
    else:
        print("FAILED: Valid layout not found.")

    # 清理数据
    HouseDeal.objects.filter(house_id__in=test_ids).delete()

if __name__ == '__main__':
    test_room_hall_nan_fix()
