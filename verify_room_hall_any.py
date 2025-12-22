
import os
import django
import json
import pandas as pd
from datetime import date

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from house_data.models import HouseDeal
from house_data.views import room_hall_any
from django.test import RequestFactory

def verify_room_hall_any():
    # 1. Setup test data
    print("Setting up test data...")
    HouseDeal.objects.filter(house_id__startswith='test_rh_').delete()
    
    deals = [
        HouseDeal(
            house_id='test_rh_1', title='T1', city='Beijing', district='Haidian', district_area='Area1',
            layout='2室1厅', area=80.0, floor='6', direction='South', deal_price=800, deal_date=date(2023, 1, 1)
        ),
        HouseDeal(
            house_id='test_rh_2', title='T2', city='Beijing', district='Haidian', district_area='Area1',
            layout='2室1厅', area=85.0, floor='6', direction='South', deal_price=850, deal_date=date(2023, 1, 2)
        ),
        HouseDeal(
            house_id='test_rh_3', title='T3', city='Beijing', district='Haidian', district_area='Area1',
            layout='3室1厅', area=120.0, floor='6', direction='South', deal_price=1200, deal_date=date(2023, 1, 3)
        ),
        HouseDeal(
            house_id='test_rh_4', title='T4', city='Beijing', district='Chaoyang', district_area='Area2',
            layout='1室1厅', area=50.0, floor='6', direction='South', deal_price=400, deal_date=date(2023, 1, 4)
        ),
    ]
    HouseDeal.objects.bulk_create(deals)
    
    factory = RequestFactory()
    
    # 2. Test District 'Haidian'
    print("\nTesting district='Haidian'...")
    request = factory.get('/house_data/room-hall-any/', {'district': 'Haidian', 'city': 'Beijing'})
    response = room_hall_any(request)
    content = json.loads(response.content.decode('utf-8'))
    print("Response for Haidian:", content)
    
    # Validation
    # 2室1厅: 2 deals, avg price (100000 + 100000)/2 = 100000. Wait:
    # Deal 1: 800w / 80m2 = 10w/m2
    # Deal 2: 850w / 85m2 = 10w/m2
    # Deal 3: 1200w / 120m2 = 10w/m2
    # So unit price should be 100000.
    
    # 3. Test District 'all'
    print("\nTesting district='all'...")
    request_all = factory.get('/house_data/room-hall-any/', {'district': 'all', 'city': 'Beijing'})
    response_all = room_hall_any(request_all)
    content_all = json.loads(response_all.content.decode('utf-8'))
    print("Response for all:", content_all)
    
    # 4. Clean up
    print("\nCleaning up...")
    HouseDeal.objects.filter(house_id__startswith='test_rh_').delete()

if __name__ == '__main__':
    verify_room_hall_any()
