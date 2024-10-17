import locale
from kafka import KafkaConsumer
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from btc_streamer.ml.model_utils import load_model, predict

# Create a consumer instance
consumer = KafkaConsumer(
    'bitcoin',
    bootstrap_servers='localhost:9092',
                      api_version=(0,11,5),

    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

# Set the locale for formatting currency values
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


timestamps = []
prices = []
base_df = pd.DataFrame()

model = load_model('models/xgboost_model_2024-10-17_10_30_57')

# Start consuming
for message in consumer:
    pred = predict(model=model, response=message.value)    
    
    pred_df = pred.toPandas()
    
    message_df = pd.DataFrame([message.value])
    message_df['prediction'] =  pred_df['prediction']
    message_df['probability'] =  pred_df['probability'].values[0][-1]
    message_df['timestamp'] = pd.to_datetime(message_df['timestamp'])
    
    print(message_df.timestamp)
    
    message_df.set_index('timestamp', inplace=True)
    
    base_df = pd.concat([message_df, base_df], axis=1)
    
    
    if len(base_df) == 5:
        import ipdb; ipdb.set_trace()    
    

    # # Round the value to the nearest dollar
    # rounded_value = round(value)

    # # Format the rounded value as a currency string without decimal places
    # formatted_value = locale.currency(rounded_value, grouping=True).replace('.00', '')

    # # Append the timestamp and price to the lists
    # timestamps.append(datetime.fromtimestamp(timestamp))
    # prices.append(rounded_value)

    # # Create a pandas DataFrame with timestamps and prices
    # df = pd.DataFrame({'Timestamp': timestamps, 'Price': prices})

    # # Calculate the 20-period moving average of the Bitcoin price
    # df['MA'] = df['Price'].rolling(window=20).mean()

    # # Clear the previous plot
    # plt.clf()

    # # Plot the Bitcoin prices and moving average over time
    # plt.plot(df['Timestamp'], df['Price'], label='Bitcoin Price')
    # plt.plot(df['Timestamp'], df['MA'], label='Moving Average (20 periods)')
    # plt.xlabel('Time')
    # plt.ylabel('Bitcoin Price (USD)')
    # plt.title('Real-Time Bitcoin Price with Moving Average')
    # plt.legend()

    # # Format the y-axis labels as currency values
    # plt.gca().get_yaxis().set_major_formatter(locale.currency)

    # # Adjust the plot margins
    # plt.gcf().autofmt_xdate()

    # # Display the plot
    # plt.pause(0.001)