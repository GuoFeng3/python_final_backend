import os
import django
import pandas as pd
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from django.test import RequestFactory
from house_data.views import total_any, add_data
from house_data.models import HouseDeal

def test_city_param():
    factory = RequestFactory()
    
    # 1. Clean up existing test data
    HouseDeal.objects.filter(house_id__startswith='test_city_').delete()
    
    # 2. Add data for Beijing
    beijing_deal = HouseDeal(
        house_id='test_city_bj_1',
        title='BJ House',
        city='Beijing',
        district='Haidian',
        district_area='Haidian-Area',
        layout='1L1B',
        area=100.0,
        floor='1',
        direction='South',
        deal_price=500, # 500万
        deal_date='2023-01-01'
    )
    beijing_deal.save()
    
    # 3. Add data for Shanghai
    shanghai_deal = HouseDeal(
        house_id='test_city_sh_1',
        title='SH House',
        city='Shanghai',
        district='Pudong',
        district_area='Pudong-Area',
        layout='1L1B',
        area=100.0,
        floor='1',
        direction='South',
        deal_price=800, # 800万
        deal_date='2023-01-01'
    )
    shanghai_deal.save()
    
    print("Added test data for Beijing and Shanghai.")
    
    # 4. Test total_any for Beijing
    request_bj = factory.get('/house_data/total_any/', {'city': 'Beijing'})
    response_bj = total_any(request_bj)
    content_bj = json.loads(response_bj.content.decode('utf-8'))
    print(f"Beijing Avg Price (should be 500): {content_bj.get('平均总价')}")
    
    # 5. Test total_any for Shanghai
    request_sh = factory.get('/house_data/total_any/', {'city': 'Shanghai'})
    response_sh = total_any(request_sh)
    content_sh = json.loads(response_sh.content.decode('utf-8'))
    print(f"Shanghai Avg Price (should be 800): {content_sh.get('平均总价')}")
    
    # 6. Test total_any for All (no city param)
    request_all = factory.get('/house_data/total_any/')
    response_all = total_any(request_all)
    content_all = json.loads(response_all.content.decode('utf-8'))
    # Average of 500 and 800 is 650 (assuming only these 2 exist in DB, but there might be real data)
    # So we just check it's not 0 and likely different if we had enough data. 
    # But since we have real data, 'all' will include them. 
    # Just checking it runs is enough, or checking count if we could.
    print(f"All Cities Avg Price: {content_all.get('平均总价')}")

    # 7. Test add_data with city param
    from django.core.files.uploadedfile import SimpleUploadedFile
    
    csv_content = "house_id,title,district,district_area,layout,area,floor,direction,deal_price,deal_unit_price,deal_date\ntest_city_upload_1,Upload House,Chaoyang,Chaoyang-Area,2L1B,90,1,South,600,66666,2023-03-01"
    file = SimpleUploadedFile("test_upload.csv", csv_content.encode('utf-8'), content_type="text/csv")
    
    request_upload = factory.post('/house_data/add_data/', {'city': 'Shenzhen', 'file': file})
    # RequestFactory doesn't automatically populate FILES from data dict in the way we want for mixed form/file
    # But passing data dict to post usually puts fields in POST and files in FILES if encoded properly.
    # However, for manual construction:
    
    response_upload = add_data(request_upload)
    print(f"Upload Status: {response_upload.status_code}")
    print(f"Upload Response: {response_upload.content.decode('utf-8')}")
    
    # Verify uploaded data has city='Shenzhen'
    uploaded_deal = HouseDeal.objects.filter(house_id='test_city_upload_1').first()
    if uploaded_deal:
        print(f"Uploaded Deal City: {uploaded_deal.city}")
    else:
        print("Uploaded deal not found!")
        
    # Clean up
    HouseDeal.objects.filter(house_id__startswith='test_city_').delete()
    print("Cleaned up test data.")

if __name__ == '__main__':
    test_city_param()
