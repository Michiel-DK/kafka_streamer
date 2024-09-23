import requests
from kafka import KafkaProducer
from time import sleep
import json
import os

# Create a producer instance
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
                  api_version=(0,11,5),

    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


def get_crypto_price():
    url = f"https://api.coinpaprika.com/v1/tickers/btc-bitcoin"
    
    headers = {
    'Accept-Encoding': 'gzip',
    'Authorization': f'Bearer {os.getenv("COINCAP_API_KEY")}',
}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    try:
        diction = data['quotes']['USD'] 
        diction['symbol'] = data['symbol']
        diction['beta_value'] = data['beta_value']
        return diction
    except:
        print(f"Error retrieving price")
        return None

while True:
    # Get Bitcoin and Ethereum data
    bitcoin_price = get_crypto_price()

    # Send Bitcoin and Ethereum data to their respective Kafka topics
    producer.send('bitcoin', value=bitcoin_price)

    # Wait for 10 seconds
    sleep(120)