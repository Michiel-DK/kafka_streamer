from btc_streamer.ml.preprocessing import BTCDataloader
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
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
    df = btc.preproc_split(df)
    
    # Convert feature columns into a single vector column
    feature_columns = [x.name for x in df.schema if re.search(r'percent', x.name)]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
        
    assembled_data = assembler.transform(df).select(['features', 'target'])
    
    # Split data into training and test sets
    train_data, test_data = assembled_data.randomSplit([0.8, 0.2], seed=42)

    
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
        num_workers=2,
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
        spark.stop()
        logging.info("Spark session stopped.")
