# Create the Spark Session
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType, ArrayType, DoubleType, FloatType, IntegerType, LongType
from pyspark.sql.functions import from_json
from pyspark.sql.functions import  col

def setup_spark():

    spark = SparkSession \
        .builder \
        .appName("btc_streamer") \
        .config("spark.streaming.stopGracefullyOnShutdown", True) \
        .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0') \
        .config("spark.sql.shuffle.partitions", 1) \
        .master("local[*]") \
        .getOrCreate()
        
    streaming_df = spark.readStream\
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "bitcoin") \
        .option("startingOffsets", "earliest") \
        .load()



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
        StructField("ath_price", DoubleType(), True),
        StructField("ath_date", StringType(), True),
        StructField("percent_from_price_ath", DoubleType(), True),
        StructField("symbol", StringType(), True),
        StructField("beta_value", DoubleType(), True)
    ])

    # Cast the value from binary to string, since Kafka sends messages as bytes
    kafka_df =streaming_df.selectExpr("CAST(value AS STRING) as json_string")

    # Parse the JSON string in the 'value' column using the defined schema
    json_df = kafka_df.withColumn("json_data", from_json(col("json_string"), schema))

    # Extract specific fields from the JSON
    extracted_df = json_df.select(
        col("json_data.price"),
        col("json_data.volume_24h_change_24h"),
        col("json_data.market_cap"),
        col("json_data.market_cap_change_24h"),
        col("json_data.percent_change_15m"),
        col("json_data.percent_change_30m"),
        col("json_data.percent_change_1h"),
        col("json_data.percent_change_6h"),
        col("json_data.percent_change_12h"),
        col("json_data.percent_change_24h"),
        col("json_data.percent_change_7d"),
        col("json_data.percent_change_30d"),
        col("json_data.percent_change_1y"),
        col("json_data.beta_value")    
    )
    
    return extracted_df

if __name__ == '__main__':
    extracted_df = setup_spark()
    # Write the output to console (for debugging) or further processing
    query = extracted_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .start()

    # Await termination of the streaming query
    query.awaitTermination()