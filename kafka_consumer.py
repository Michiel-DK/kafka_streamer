import locale
from kafka import KafkaConsumer
import json
from datetime import datetime

# Create a consumer instance
consumer = KafkaConsumer(
    'bitcoin',
    bootstrap_servers='localhost:9092',
                      api_version=(0,11,5),

    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

# Start consuming
for message in consumer:
     print(message)