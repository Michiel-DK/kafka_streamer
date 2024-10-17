from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType
from pyspark.sql.functions import col, sum as _sum, last, window, from_unixtime, lag, round, when, lit
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F



import os
import re

class BTCDataloader():
    
    def __init__(self):
        self.spark = None
        self.feature_columns = None
        self.test_set = None
        
    def setup_spark(self, app_name:str= 'CryptoDataProcessing',  memory:str = '2g', partitions:str = '4'):
        
        self.spark = SparkSession.builder.appName(app_name)\
            .config("spark.executor.memory", memory).config("spark.sql.shuffle.partitions", partitions)\
                .getOrCreate()
        
    
    def load_data(self, directory:str = 'data/'):
        
        file_list = os.listdir(directory)

        # Regular expression to match a year (four consecutive digits)
        year_pattern = re.compile(r'\d{4}')

        # Filter files that contain a year in their names
        files_with_years = sorted([directory + file for file in file_list if year_pattern.search(file)])
        
        df = self.spark.read.csv(files_with_years, header=True, inferSchema=True)
        
        return df
    
    def preproc_split(self, df, drop:bool = True, split:bool = True):
        
        # Step 1: Convert unix time to timestamp
        df = df.withColumn("timestamp", from_unixtime(col("unix")).cast(TimestampType())).orderBy(col("timestamp")).drop('date')

        # Step 2: Resample the DataFrame to 5-minute intervals
        df_5min = df.groupBy(window(col("timestamp"), "5 minutes").alias("time_window")) \
            .agg(
                last("close").alias("price"),          # Get the last close price within the 5-minute window
                _sum("Volume USD").alias("volume_5m")  # Sum of Volume USD for the 5-minute window
            ).orderBy(col("time_window.end"))

        # Step 4: Select the desired columns and rename window to timestamp
        df_5min = df_5min.select(
            col("time_window.end").alias("timestamp"),  # Use the end of the 5-minute window as the timestamp
            col("price"),
            col("volume_5m").alias("Volume_USD"),
        )
        
        
        window_spec = Window.partitionBy(window(col("timestamp"), "1 week").alias("month")).orderBy("timestamp")

        #create lagged columns
        df_transformed = df_5min.withColumn(
            "percent_change_15m", 
            (col("price") - lag("price", 3).over(window_spec)) / lag("price", 3).over(window_spec)
        ).withColumn(
            "percent_change_30m", 
            (col("price") - lag("price", 6).over(window_spec)) / lag("price", 6).over(window_spec)
        ).withColumn(
            "percent_change_1h", 
            (col("price") - lag("price", 12).over(window_spec)) / lag("price", 12).over(window_spec)
        ).withColumn(
            "percent_change_6h", 
            (col("price") - lag("price", 72).over(window_spec)) / lag("price", 72).over(window_spec)
        ).withColumn(
            "percent_change_12h", 
            (col("price") - lag("price", 144).over(window_spec)) / lag("price", 144).over(window_spec)
        ).withColumn(
            "percent_change_24h", 
            (col("price") - lag("price", 288).over(window_spec)) / lag("price", 288).over(window_spec)
        )
        
        if drop:
            df_final = df_transformed.na.drop()
                
        #create future column for target
        df_final = df_final.withColumn("price_plus_15m", F.lag("price", -3).over(window_spec))
        
        df_final = df_final.withColumn("price_change_15m", (F.col("price_plus_15m") - F.col("price")) / F.col("price"))
        
        df_final = df_final.withColumn(
            "target",
            when(col("price_change_15m") > 0, 1).otherwise(0)
        )
        
        train_set = df_final.filter(F.year(F.col('timestamp')) != 2023)  # All other years
        test_set = df_final.filter(F.year(F.col('timestamp')) == 2023)   # Only 2023 observations
        
        self.test_set = test_set
        
        # Convert feature columns into a single vector column
        self.feature_columns = [x.name for x in train_set.schema if re.search(r'percent', x.name)]
        assembler = VectorAssembler(inputCols=self.feature_columns, outputCol='features')
            
        assembled_data_train = assembler.transform(train_set).select(['features', 'target'])
        assembled_data_test = assembler.transform(test_set).select(['features', 'target'])
        
        if split:
        
            #df_pipe, df_pipe2 = assembled_data_train.randomSplit(weights=[0.2, 0.8], seed=42)
            df_pipe = assembled_data_train
            
            # Split data into training and test sets
            train_data, val_data = df_pipe.randomSplit([0.8, 0.2], seed=42)
            
            train_data = train_data.withColumn('isVal', lit(False))
            val_data = val_data.withColumn('isVal', lit(True))
            
            train_data = train_data.union(val_data)
                        
            print(f"train: {train_data.count()}, test: {assembled_data_test.count()}")

            return train_data, assembled_data_test, self.spark

        else:
            return None
    
if __name__=='__main__':
    try:
        btc = BTCDataloader()
        btc.setup_spark()
        df = btc.load_data()
        train_data, test_data, preproc_spark = btc.preproc_split(df, split=True)
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)   