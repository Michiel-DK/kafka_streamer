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
from pyspark.sql import functions as F
from btc_streamer.utils import *

import wandb



# Initialize logging
logging.basicConfig(level=logging.INFO)



wandb.init(project="xg_boost_test")



class XGBoostTrainer():
    def __init__(self) -> None:
        self.spark = None
        self.app_name = None
    
    def setup_spark(self, app_name: 'str' = 'XGBoostSpark'):
        self.app_name = app_name
        self.spark = SparkSession.builder \
            .appName(self.app_name) \
            .getOrCreate()
        logging.info("Spark session initialized successfully.")
    
    def train_model(self, train_data, xgb_estimator=None):
        
        if not xgb_estimator:
            # Create and train the XGBoostEstimator model
            xgb_estimator = SparkXGBClassifier(
                features_col='features',
                label_col='target',
                num_workers=4,
                device='cpu',
                validation_indicator_col = 'isVal',
                eval_metric='auc',
            )
        
        xgb_estimator.setParams(early_stopping_rounds=10)
        
        self.model = xgb_estimator.fit(train_data)
        
        return self.model
    
    def score_model(self, test_data, feature_columns, original_test_set, model=None):
        
        # Make predictions
        if model is None:
            predictions = self.model.transform(test_data)
        else:
            self.model = model
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
        preds = preds.withColumn("index", F.monotonically_increasing_id())
        
        original_test_set = original_test_set.withColumn("index", F.monotonically_increasing_id())
        
        plot_set = np.array(preds.join(original_test_set, on="index", how="outer").select("prediction", 'index_1_probability', 'price', 'timestamp').collect())
    
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
                
        sklearn_model = self.model._xgb_sklearn_model
        
        wandb.log(scores)
        
        wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels=['sell', 'buy'])
        
        wandb.sklearn.plot_roc(y_true, y_prob, labels=['sell', 'buy'])
        
        wandb.sklearn.plot_precision_recall(y_true, y_prob, labels=['sell', 'buy'])
        
        wandb.sklearn.plot_feature_importances(sklearn_model, feature_columns)
        
        bokeh_file_name = plot_blokeh(plot_set)
        bokeh_html = wandb.Html(bokeh_file_name)
        pred_plot = wandb.Table(columns=["prediction plot"], data=[[bokeh_html]])
        wandb.log({"prediction_table": pred_plot})
        
        plotly_file_name = plot_probas(plot_set)
        plotly_html = wandb.Html(plotly_file_name)
        proba_plot = wandb.Table(columns=["proba plot"], data=[[plotly_html]])
        wandb.log({"proba_table": proba_plot})
        
        wandb.finish()

        self.spark.stop()
        logging.info("Spark session stopped.")

if __name__=='__main__':
    try:
        btc = BTCDataloader()
        btc.setup_spark()
        df = btc.load_data()
        train_data, test_data, preproc_spark = btc.preproc_split(df)
        
        
        xg = XGBoostTrainer()
        xg.setup_spark()
        
        
        model = xg.train_model(train_data)
        xg.score_model(test_data, btc.feature_columns, btc.test_set)
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)