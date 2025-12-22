import os
import django
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonfinal.settings')
django.setup()

from django.test import RequestFactory
from house_data.views import predict_price

def test_view():
    factory = RequestFactory()
    # Test with '昌平'
    request = factory.get('/house_data/predict_price/', {'行政区名': '昌平'})
    response = predict_price(request)
    
    print(f"Status Code: {response.status_code}")
    print("Content:")
    # Parse JSON content
    content = json.loads(response.content.decode('utf-8'))
    print(json.dumps(content, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    test_view()
