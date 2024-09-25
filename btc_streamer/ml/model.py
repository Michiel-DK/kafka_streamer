from btc_streamer.ml.preprocessing import BTCDataloader
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from xgboost.spark import SparkXGBClassifier
import re
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col




# Initialize logging
logging.basicConfig(level=logging.INFO)

import wandb

wandb.init(project="xg_boost_test")

class XGBoostTrainer():
    def __init__(self) -> None:
        pass
    
    def setup_spark(self):
        self.spark = SparkSession.builder \
            .appName("XGBoostSpark") \
            .getOrCreate()
        logging.info("Spark session initialized successfully.")
        
    def get_data(self):
        btc = BTCDataloader()
        btc.setup_spark()
        df = btc.load_data()
        train_data, test_data, preproc_spark = btc.preproc_split(df)
        
        return train_data, test_data
    
    def train_model(self, train_data):
        
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
            params=params,
            num_workers=4,
            device='cpu'
        )
        
        self.model = xgb_estimator.fit(train_data)
        
        return self.model
    
    def score_model(self, test_data):
        
        # Make predictions
        predictions = self.model.transform(test_data)
        
        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(
            labelCol='target',
            rawPredictionCol='rawPrediction',
            metricName='areaUnderROC'
        )

        roc_auc = evaluator.evaluate(predictions)
        
        preds = self.model.transform(test_data).select("target", "prediction", 'probability')
                
        preds = preds.withColumn("probability_array", vector_to_array(col("probability")))
        preds = preds.withColumn("index_0_probability", col("probability_array")[0])
        preds = preds.withColumn("index_1_probability", col("probability_array")[1])

        preds_numpy = np.array(preds.select("target", "prediction", 'index_0_probability','index_1_probability').collect())
        
        y_true = preds_numpy[:,0]
        y_pred = preds_numpy[:,1]
        y_prob = preds_numpy[:,-2:]
                
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        scores = {
            'roc_auc': roc_auc,
            'accuracy': acc,
            'precision': precision,
            'recall':recall
        }
        
        wandb.log(scores)
        
        wandb.sklearn.plot_confusion_matrix(y_true, y_pred)
        
        wandb.sklearn.plot_roc(y_true, y_prob, labels=['sell', 'buy'])
        
        wandb.sklearn.plot_precision_recall(y_true, y_prob, labels=['sell', 'buy'])
        
        self.spark.stop()
        logging.info("Spark session stopped.")

if __name__=='__main__':
    try:
        xg = XGBoostTrainer()
        xg.setup_spark()
        train, test = xg.get_data()
        model = xg.train_model(train)
        xg.score_model(test)
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)