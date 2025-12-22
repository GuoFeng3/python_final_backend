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
    
    # Create a small dummy CSV file in memory for testing
    csv_content = """house_id,title,district,district_area,layout,area,floor,direction,deal_price,deal_unit_price,deal_date
test_001,测试房屋1,测试区,测试区-测试地,1室1厅,50.00㎡,共6层,南,200万,,2025-01-01
test_002,测试房屋2,测试区,测试区-测试地,2室1厅,80.00㎡,共6层,南北,300万,,2025-02-01
"""
    
    from django.core.files.uploadedfile import SimpleUploadedFile
    file = SimpleUploadedFile("test_data.csv", csv_content.encode('utf-8'), content_type="text/csv")
    
    print("Testing upload of CSV data...")
    request = factory.post('/house_data/add_data/', {'file': file})
    response = add_data(request)
    
    print(f"Status Code: {response.status_code}")
    content = json.loads(response.content.decode('utf-8'))
    print(json.dumps(content, ensure_ascii=False, indent=4))
    
    # Verify data in database
    from house_data.models import HouseDeal
    count = HouseDeal.objects.filter(house_id__startswith='test_').count()
    print(f"Found {count} test records in database.")
    
    # Clean up test data
    deleted_count, _ = HouseDeal.objects.filter(house_id__startswith='test_').delete()
    print(f"Cleaned up {deleted_count} test records.")

if __name__ == '__main__':
    test_view()
