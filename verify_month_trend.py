import os
import django
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from django.test import RequestFactory
from house_data.views import month_trend

def test_view():
    factory = RequestFactory()
    # Test with '昌平'
    request = factory.get('/house_data/month-trend/', {'行政区名': '海淀'})
    response = month_trend(request)
    
    print(f"Status Code: {response.status_code}")
    print("Content:")
    # Parse JSON content
    content = json.loads(response.content.decode('utf-8'))
    # Print first 5 records to verify
    print(json.dumps(content, ensure_ascii=False, indent=4))
    
    print(f"Total records: {len(content)}")

if __name__ == '__main__':
    test_view()
