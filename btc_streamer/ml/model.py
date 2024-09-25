from btc_streamer.ml.preprocessing import BTCDataloader
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from xgboost.spark import SparkXGBClassifier
import re
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

try:
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("XGBoostSparkTest") \
        .getOrCreate()
    logging.info("Spark session initialized successfully.")
    
    # Load and prepare data
    btc = BTCDataloader()
    btc.setup_spark()
    df = btc.load_data()
    train_data, test_data, preproc_spark = btc.preproc_split(df)
    
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'booster': 'gbtree',
        'tree_method': 'auto',           # Ensure using CPU
        'num_round': 100,
        'eta': 0.1
    }
    
    # Create and train the XGBoostEstimator model
    xgb_estimator = SparkXGBClassifier(
        features_col='features',
        label_col='target',
        #params=params,
        num_workers=4,
        device='cpu'
    )
    
    model = xgb_estimator.fit(train_data)
    logging.info("Model training completed.")
    
    # Make predictions
    predictions = model.transform(test_data)
    
    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(
        labelCol='target',
        rawPredictionCol='rawPrediction',
        metricName='areaUnderROC'
    )
    
    roc_auc = evaluator.evaluate(predictions)
    print(f'Area Under ROC: {roc_auc}')
    
except Exception as e:
    logging.error("An error occurred:", exc_info=True)

finally:
    # Stop Spark session
    if 'spark' in locals():
        preproc_spark.stop()
        spark.stop()
        logging.info("Spark session stopped.")
