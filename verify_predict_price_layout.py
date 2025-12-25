
import os
import django
import json
import pandas as pd
from datetime import date

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from house_data.models import HouseDeal
from house_data.views import predict_price
from django.test import RequestFactory

def verify_predict_price_layout():
    # 1. Setup test data
    print("Setting up test data...")
    HouseDeal.objects.filter(house_id__startswith='test_pp_').delete()
    
    # Create data spanning multiple months for regression
    # Layout '2室1厅' - increasing prices
    deals_2room = [
        HouseDeal(house_id='test_pp_1', title='T1', city='Beijing', district='Haidian', district_area='A', layout='2室1厅', area=100, floor='6', direction='S', deal_price=500, deal_date=date(2023, 1, 15)), # 5w/m2
        HouseDeal(house_id='test_pp_2', title='T2', city='Beijing', district='Haidian', district_area='A', layout='2室1厅', area=100, floor='6', direction='S', deal_price=600, deal_date=date(2023, 2, 15)), # 6w/m2
        HouseDeal(house_id='test_pp_3', title='T3', city='Beijing', district='Haidian', district_area='A', layout='2室1厅', area=100, floor='6', direction='S', deal_price=700, deal_date=date(2023, 3, 15)), # 7w/m2
    ]
    
    # Layout '1室1厅' - decreasing prices
    deals_1room = [
        HouseDeal(house_id='test_pp_4', title='T4', city='Beijing', district='Haidian', district_area='A', layout='1室1厅', area=50, floor='6', direction='S', deal_price=500, deal_date=date(2023, 1, 15)), # 10w/m2
        HouseDeal(house_id='test_pp_5', title='T5', city='Beijing', district='Haidian', district_area='A', layout='1室1厅', area=50, floor='6', direction='S', deal_price=400, deal_date=date(2023, 2, 15)), # 8w/m2
        HouseDeal(house_id='test_pp_6', title='T6', city='Beijing', district='Haidian', district_area='A', layout='1室1厅', area=50, floor='6', direction='S', deal_price=300, deal_date=date(2023, 3, 15)), # 6w/m2
    ]
    
    HouseDeal.objects.bulk_create(deals_2room + deals_1room)
    
    factory = RequestFactory()
    
    # 2. Test layout='2室1厅'
    print("\nTesting layout='2室1厅' (Expected increasing trend)...")
    request_2room = factory.get('/house_data/predict-price/', {'district': 'Haidian', 'city': 'Beijing', 'layout': '2室1厅'})
    response_2room = predict_price(request_2room)
    content_2room = json.loads(response_2room.content.decode('utf-8'))
    print("Response for 2室1厅:", content_2room)
    # Expected: Next month should be > 70000 (around 80000)
    
    # 3. Test layout='1室1厅'
    print("\nTesting layout='1室1厅' (Expected decreasing trend)...")
    request_1room = factory.get('/house_data/predict-price/', {'district': 'Haidian', 'city': 'Beijing', 'layout': '1室1厅'})
    response_1room = predict_price(request_1room)
    content_1room = json.loads(response_1room.content.decode('utf-8'))
    print("Response for 1室1厅:", content_1room)
    # Expected: Next month should be < 60000 (around 40000)
    
    # 4. Test layout='all' (or missing)
    print("\nTesting layout='all' (Mixed trend)...")
    request_all = factory.get('/house_data/predict-price/', {'district': 'Haidian', 'city': 'Beijing'})
    response_all = predict_price(request_all)
    content_all = json.loads(response_all.content.decode('utf-8'))
    print("Response for all:", content_all)
    
    # 5. Clean up
    print("\nCleaning up...")
    HouseDeal.objects.filter(house_id__startswith='test_pp_').delete()

if __name__ == '__main__':
    verify_predict_price_layout()
