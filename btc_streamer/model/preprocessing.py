from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType
from pyspark.sql.functions import col, lag, from_unixtime, round
from pyspark.sql.window import Window

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("CryptoDataProcessing") \
    .getOrCreate()

# Step 2: Define the schema
schema = StructType([
    StructField("price", DoubleType(), True),
    StructField("volume_24h", DoubleType(), True),
    StructField("volume_24h_change_24h", DoubleType(), True),
    StructField("market_cap", LongType(), True),
    StructField("market_cap_change_24h", DoubleType(), True),
    StructField("percent_change_15m", DoubleType(), True),
    StructField("percent_change_30m", DoubleType(), True),
    StructField("percent_change_1h", DoubleType(), True),
    StructField("percent_change_6h", DoubleType(), True),
    StructField("percent_change_12h", DoubleType(), True),
    StructField("percent_change_24h", DoubleType(), True),
    StructField("percent_change_7d", DoubleType(), True),
    StructField("percent_change_30d", DoubleType(), True),
    StructField("percent_change_1y", DoubleType(), True),
    StructField("symbol", StringType(), True),
    StructField("beta_value", DoubleType(), True),
    StructField("timestamp", TimestampType(), True)
])

# Step 3: Load the CSV file into a DataFrame
file_path = "path_to_your_file.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Step 4: Window specification for calculating percentage changes
window_spec = Window.orderBy("unix")

# Helper function to calculate percentage change
def calculate_percentage_change(current, previous):
    return (current - previous) / previous * 100

# Step 5: Transform the DataFrame according to the schema
df_transformed = df.select(
    col("close").alias("price"),
    col("Volume USD").alias("volume_24h"),

    # Calculate volume_24h_change_24h as the percentage change from the previous row
    round((col("Volume USD") - lag("Volume USD", 1).over(window_spec)) / lag("Volume USD", 1).over(window_spec) * 100, 2)
    .alias("volume_24h_change_24h"),
    
    # Placeholder for market_cap and market_cap_change_24h
    col("Volume USD").cast(LongType()).alias("market_cap"),
    col("Volume USD").alias("market_cap_change_24h"),
    
    # Calculate percentage change for different intervals (15m, 30m, 1h, 6h, 12h, 24h, 7d, 30d, 1y)
    round(calculate_percentage_change(col("close"), lag("close", 3).over(window_spec)), 2).alias("percent_change_15m"),  # Assuming 1m intervals
    round(calculate_percentage_change(col("close"), lag("close", 6).over(window_spec)), 2).alias("percent_change_30m"),
    round(calculate_percentage_change(col("close"), lag("close", 12).over(window_spec)), 2).alias("percent_change_1h"),
    round(calculate_percentage_change(col("close"), lag("close", 72).over(window_spec)), 2).alias("percent_change_6h"),   # 72 mins is approx. 6h
    round(calculate_percentage_change(col("close"), lag("close", 144).over(window_spec)), 2).alias("percent_change_12h"), # 144 mins is approx. 12h
    round(calculate_percentage_change(col("close"), lag("close", 288).over(window_spec)), 2).alias("percent_change_24h"), # 288 mins is approx. 24h
    round(calculate_percentage_change(col("close"), lag("close", 2016).over(window_spec)), 2).alias("percent_change_7d"), # 2016 mins is approx. 7 days
    round(calculate_percentage_change(col("close"), lag("close", 8640).over(window_spec)), 2).alias("percent_change_30d"),# 8640 mins is approx. 30 days
    round(calculate_percentage_change(col("close"), lag("close", 105120).over(window_spec)), 2).alias("percent_change_1y"),# 105120 mins is approx. 1 year
    
    # Map the `symbol` from CSV
    col("symbol"),
    
    # Placeholder for beta_value
    col("Volume USD").alias("beta_value"),
    
    # Convert unix timestamp to actual timestamp
    from_unixtime(col("unix")).alias("timestamp")
)

# Step 6: Show the transformed DataFrame
df_transformed.show()
