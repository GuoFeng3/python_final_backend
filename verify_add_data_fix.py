import os
import django
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from django.test import RequestFactory
from house_data.views import add_data

def test_view():
    factory = RequestFactory()
    
    # Create a small dummy CSV file in memory for testing with non-standard date format
    csv_content = """house_id,title,district,district_area,layout,area,floor,direction,deal_price,deal_unit_price,deal_date
test_fix_001,测试房源,平谷,平谷-马坊,3室2厅,101.83㎡,共17层,南,225万,,2023/6/7
test_fix_002,测试房源2,平谷,平谷-马坊,2室1厅,80.00㎡,共6层,南,200万,,2023.06.08
test_fix_003,测试房源3,平谷,平谷-马坊,1室1厅,50.00㎡,共6层,南,150万,,2023-06-09
"""
    
    from django.core.files.uploadedfile import SimpleUploadedFile
    file = SimpleUploadedFile("test_data_fix.csv", csv_content.encode('utf-8'), content_type="text/csv")
    
    print("Testing upload of CSV data with various date formats...")
    request = factory.post('/house_data/add_data/', {'file': file})
    response = add_data(request)
    
    print(f"Status Code: {response.status_code}")
    content = json.loads(response.content.decode('utf-8'))
    print(json.dumps(content, ensure_ascii=False, indent=4))
    
    # Verify data in database
    from house_data.models import HouseDeal
    count = HouseDeal.objects.filter(house_id__startswith='test_fix_').count()
    print(f"Found {count} test records in database.")
    
    # Check specific dates
    record1 = HouseDeal.objects.filter(house_id='test_fix_001').first()
    if record1:
        print(f"Record 1 Date: {record1.deal_date} (Expected: 2023-06-07)")
        
    record2 = HouseDeal.objects.filter(house_id='test_fix_002').first()
    if record2:
        print(f"Record 2 Date: {record2.deal_date} (Expected: 2023-06-08)")

    # Clean up test data
    deleted_count, _ = HouseDeal.objects.filter(house_id__startswith='test_fix_').delete()
    print(f"Cleaned up {deleted_count} test records.")

if __name__ == '__main__':
    test_view()
