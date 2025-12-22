import os
import django
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from django.test import RequestFactory
from house_data.views import squaremeter_avgprice

def test_view():
    factory = RequestFactory()
    
    # Test case 1: Specific district
    print("Testing with district='昌平'...")
    request = factory.get('/house_data/squaremeter-avgprice/', {'行政区名': '昌平'})
    response = squaremeter_avgprice(request)
    print(f"Status Code: {response.status_code}")
    content = json.loads(response.content.decode('utf-8'))
    print(json.dumps(content, ensure_ascii=False, indent=4))
    
    print("-" * 50)
    
    # Test case 2: All districts
    print("Testing with district='all'...")
    request_all = factory.get('/house_data/squaremeter-avgprice/', {'行政区名': 'all'})
    response_all = squaremeter_avgprice(request_all)
    print(f"Status Code: {response_all.status_code}")
    content_all = json.loads(response_all.content.decode('utf-8'))
    # Show first 2 items to avoid clutter
    print(json.dumps(content_all, ensure_ascii=False, indent=4)) 

if __name__ == '__main__':
    test_view()
