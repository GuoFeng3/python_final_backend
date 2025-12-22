import os
import django
import pandas as pd
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from django.test import RequestFactory
from house_data.views import total_any
from house_data.models import HouseDeal

def test_total_any_district():
    factory = RequestFactory()
    
    # 1. Clean up existing test data
    HouseDeal.objects.filter(house_id__startswith='test_ta_').delete()
    
    # 2. Add data for District A (Haidian)
    deal_a = HouseDeal(
        house_id='test_ta_1',
        title='Haidian House',
        city='Beijing',
        district='Haidian',
        district_area='Haidian-Area',
        layout='1L1B',
        area=100.0,
        floor='1',
        direction='South',
        deal_price=1000, # 1000万, Unit Price 100000
        deal_date='2023-01-01'
    )
    deal_a.save()
    
    # 3. Add data for District B (Chaoyang)
    deal_b = HouseDeal(
        house_id='test_ta_2',
        title='Chaoyang House',
        city='Beijing',
        district='Chaoyang',
        district_area='Chaoyang-Area',
        layout='1L1B',
        area=100.0,
        floor='1',
        direction='South',
        deal_price=500, # 500万, Unit Price 50000
        deal_date='2023-01-01'
    )
    deal_b.save()
    
    print("Added test data for Haidian (1000) and Chaoyang (500).")
    
    # 4. Test total_any with no district (should be all -> avg 750)
    # Note: Existing DB data might affect this if we don't filter carefully.
    # But since we added test data, let's hope they are distinct or we can filter by city/district to check logic.
    # To be precise, let's filter by city='Beijing' and hope other tests didn't leave too much mess, 
    # OR we can just check if logic works.
    # The best way is to use a unique city for test? Or just trust the logic if values differ.
    
    # Let's try requesting specifically for Haidian
    request_haidian = factory.get('/house_data/total_any/', {'district': 'Haidian'})
    response_haidian = total_any(request_haidian)
    content_haidian = json.loads(response_haidian.content.decode('utf-8'))
    print(f"Haidian Response: {content_haidian}")
    
    # Request for Chaoyang
    request_chaoyang = factory.get('/house_data/total_any/', {'district': 'Chaoyang'})
    response_chaoyang = total_any(request_chaoyang)
    content_chaoyang = json.loads(response_chaoyang.content.decode('utf-8'))
    print(f"Chaoyang Response: {content_chaoyang}")
    
    # Request for All (explicit)
    request_all_explicit = factory.get('/house_data/total_any/', {'district': 'all'})
    response_all_explicit = total_any(request_all_explicit)
    content_all_explicit = json.loads(response_all_explicit.content.decode('utf-8'))
    print(f"All (Explicit) Avg Total Price: {content_all_explicit.get('平均总价')}")

    # Request for All (implicit)
    request_all_implicit = factory.get('/house_data/total_any/')
    response_all_implicit = total_any(request_all_implicit)
    content_all_implicit = json.loads(response_all_implicit.content.decode('utf-8'))
    print(f"All (Implicit) Avg Total Price: {content_all_implicit.get('平均总价')}")
    
    # Clean up
    HouseDeal.objects.filter(house_id__startswith='test_ta_').delete()
    print("Cleaned up test data.")

if __name__ == '__main__':
    test_total_any_district()
