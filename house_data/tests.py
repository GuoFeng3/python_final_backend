from django.test import TestCase, Client
from django.urls import reverse
from .models import HouseDeal
from datetime import date
import json

class PredictPriceTest(TestCase):
    def setUp(self):
        self.client = Client()
        for i in range(36):
            year = 2023 + (i // 12)
            month = 1 + (i % 12)
            deal_price = 100 + i * 20
            HouseDeal.objects.create(
                house_id=str(i + 1),
                title=f't{i + 1}',
                deal_price=deal_price,
                area=100.0,
                deal_date=date(year, month, 1),
                district='test_dist',
            )
        
    def test_predict_price_linear(self):
        url = reverse('predict_price_linear')
        response = self.client.get(url, {'district': 'test_dist'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('history', data)
        self.assertIn('prediction', data)
        self.assertEqual(len(data['prediction']), 6)

        self.assertEqual(data['prediction'][0]['month'], '2026-01')
        self.assertIn('price', data['prediction'][0])
        self.assertAlmostEqual(float(data['prediction'][0]['price']), 82000.0, delta=500.0)

    def test_predict_price_rnn(self):
        url = reverse('predict_price_rnn')
        response = self.client.get(url, {'district': 'test_dist'})
        
        if response.status_code != 200:
            print(f"RNN Error: {response.content.decode('utf-8')}")
            
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('history', data)
        self.assertIn('prediction', data)
        self.assertEqual(len(data['prediction']), 6)
        
        self.assertEqual(data['prediction'][0]['month'], '2026-01')
        
        pred = data['prediction'][0]['price']
        self.assertGreater(pred, 0)
