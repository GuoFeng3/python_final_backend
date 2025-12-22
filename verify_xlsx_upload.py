import os
import django
import pandas as pd
import io
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from django.test import RequestFactory
from house_data.views import add_data
from house_data.models import HouseDeal

def test_xlsx_view():
    print("Checking for openpyxl...")
    try:
        import openpyxl
        print("openpyxl is installed.")
    except ImportError:
        print("openpyxl is NOT installed. Installing...")
        os.system("pip install openpyxl")
        
    factory = RequestFactory()
    
    # Create a dummy DataFrame and save to Excel
    data = {
        'house_id': ['test_xlsx_001', 'test_xlsx_002'],
        'title': ['测试XLSX房源1', '测试XLSX房源2'],
        'district': ['海淀', '海淀'],
        'district_area': ['海淀-中关村', '海淀-五道口'],
        'layout': ['2室1厅', '1室0厅'],
        'area': [88.5, 40.2],  # Float values
        'floor': ['共10层', '共6层'],
        'direction': ['南', '北'],
        'deal_price': [800, 450], # Int/Float values
        'deal_unit_price': [90000, 110000],
        'deal_date': ['2023-01-01', '2023/02/02']
    }
    df = pd.DataFrame(data)
    
    # Save to bytes buffer
    excel_file = io.BytesIO()
    df.to_excel(excel_file, index=False)
    excel_file.seek(0)
    
    from django.core.files.uploadedfile import SimpleUploadedFile
    # Name must end with .xlsx
    file = SimpleUploadedFile("test_data.xlsx", excel_file.getvalue(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    print("Testing upload of XLSX data...")
    request = factory.post('/house_data/add_data/', {'file': file})
    
    response = add_data(request)
    
    print(f"Status Code: {response.status_code}")
    content = json.loads(response.content.decode('utf-8'))
    print(json.dumps(content, ensure_ascii=False, indent=4))
    
    # Verify data in database
    count = HouseDeal.objects.filter(house_id__startswith='test_xlsx_').count()
    print(f"Found {count} test records in database.")
    
    # Check specific record
    record = HouseDeal.objects.filter(house_id='test_xlsx_001').first()
    if record:
        print(f"Record 1: Area={record.area}, Price={record.deal_price}, Date={record.deal_date}")

    # Clean up test data
    deleted_count, _ = HouseDeal.objects.filter(house_id__startswith='test_xlsx_').delete()
    print(f"Cleaned up {deleted_count} test records.")


if __name__ == '__main__':
    test_xlsx_view()
