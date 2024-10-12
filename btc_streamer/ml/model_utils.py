from xgboost.spark import SparkXGBClassifierModel
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import col, sum as _sum, last, window, from_unixtime, lag, round, when, lit
import re
from pyspark.ml.feature import VectorAssembler

def load_model(path: str = 'models/xgboost_model', session_name: str = 'XGBoostLoadModel') -> SparkXGBClassifierModel:
    """
    Loads a previously saved XGBoost model from the specified path.

    Args:
        path (str): The local path where the model is saved. Defaults to './xgboost_model'.

    Returns:
        SparkXGBClassifier: The loaded XGBoost model.
    """
    spark = SparkSession.builder.appName(session_name).getOrCreate()
    model = SparkXGBClassifierModel.load(path)
    spark.stop()
    
    return model

def predict(response: dict, model: SparkXGBClassifierModel, session_name: str = 'XGBoostPredict') -> dict:
    """
    Predicts the label for the input features using the provided XGBoost model.

    Args:
        response (dict): The input features for prediction.
        model (SparkXGBClassifierModel): The pretrained XGBoost model.

    Returns:
        dict: The response dictionary containing the input features and the predicted label.
    """



    # Initialize a Spark session
    spark = SparkSession.builder.appName(session_name).getOrCreate()

    # Creating a Pandas DataFrame to filter the required fields
    df = pd.DataFrame([response])

    # Filtering only the required percentage change columns
    filtered_df = df[['percent_change_15m', 'percent_change_30m', 'percent_change_1h', 
                    'percent_change_6h', 'percent_change_12h', 'percent_change_24h']]

    # Converting the Pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(filtered_df)

    df_final = spark_df.withColumn(
                "target",
                when(col("percent_change_15m") > 0, 1).otherwise(0)
            ).drop('percent_change_15m')

    feature_columns = [x.name for x in df_final.schema if re.search(r'percent', x.name)]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
                
    assembled_data_train = assembler.transform(df_final).select(['features', 'target'])
    
    response = model.transform(assembled_data_train).show()

    return response