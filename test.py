
import requests
import os

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
url = 'https://api.polygon.io/v2/aggs/ticker/T-BILL-13W/prev'
params = {
    'adjusted': 'true',
    'apiKey': API_KEY
}

response = requests.get(url, params=params)
data = response.json()
print([i for i in data])
#risk_free_rate = data['results'][0]['c'] / 100  # Convert from percent to decimal
