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

from datetime import datetime


import wandb
import os



# Initialize logging
logging.basicConfig(level=logging.INFO)



wandb.init(project="xg_boost_test")



class XGBoostTrainer():
    
    """
    A class to handle the training, scoring, and evaluation of an XGBoost model using Spark.
    This class initializes a Spark session, sets up an XGBoost model, and handles the full training
    and evaluation workflow for binary classification.
    """
    
    def __init__(self) -> None:
        
        """
        Initializes the XGBoostTrainer class.
        Sets the Spark session and application name to None initially.
        """
        
        self.spark = None
        self.app_name = None
        self.model = None

    
    def setup(self, app_name: 'str' = 'XGBoostSpark', model=None) -> SparkXGBClassifier:
        
        """
        Initializes the Spark session and creates an XGBoost model if not provided.

        Args:
            app_name (str): The name of the Spark application. Defaults to 'XGBoostSpark'.
            model (SparkXGBClassifier, optional): Pretrained model. If None, initializes a new model.

        Returns:
            SparkXGBClassifier: The initialized or provided XGBoost model.
        """
        
        self.app_name = app_name
        self.spark = SparkSession.builder \
            .appName(self.app_name) \
            .getOrCreate()
        logging.info("Spark session initialized successfully.")
        
        if not model:
            # Create and train the XGBoostEstimator model
            self.model = SparkXGBClassifier(
                features_col='features',
                label_col='target',
                num_workers=4,
                device='cpu',
                validation_indicator_col = 'isVal',
                eval_metric='auc',
            )
        else:
            self.model = model
            
        return self.model
    
    def train_model(self, train_data, xgb_estimator=None) -> SparkXGBClassifier:
        
        """
        Trains the XGBoost model using the provided training data.

        Args:
            train_data (DataFrame): The training dataset in Spark DataFrame format.
            xgb_estimator (SparkXGBClassifier, optional): If provided, uses the passed XGBoost estimator for training.

        Returns:
            SparkXGBClassifier: The trained XGBoost model.
        """
        
        xgb_estimator.setParams(early_stopping_rounds=10)
        
        self.model = xgb_estimator.fit(train_data)
        
        return self.model
    
    def score_model(self, train_data, test_data, feature_columns, original_test_set, best_model=None):
        
        """
        Scores and evaluates the model, logs metrics and plots using wandb.

        Args:
            train_data (DataFrame): The training dataset.
            test_data (DataFrame): The test dataset.
            feature_columns (list): List of feature column names used in training.
            original_test_set (DataFrame): The original test set for comparison.
            best_model (SparkXGBClassifier, optional): If provided, evaluates using the given model.
        """
        
        if best_model:
            self.model = best_model

        # Extract model parameters for logging
        best_params = {param.name for param in self.model.extractParamMap().keys()}
        
        param_dict = {}
        
        for param in best_params:
            value = self.model.getOrDefault(param)
            
            if value:
                param_dict[param] = value
        
        wandb.log(param_dict)
        
        # Make predictions on both train and test data
        predictions = self.model.transform(test_data)
        predictions_train = self.model.transform(train_data)
        
        # Initialize the evaluator for AUC-ROC metric
        evaluator = BinaryClassificationEvaluator(
            labelCol='target',
            rawPredictionCol='rawPrediction',
            metricName='areaUnderROC'
        )

        # Evaluate ROC for training and test data
        train_roc = evaluator.evaluate(predictions_train)
        test_roc = evaluator.evaluate(predictions)    

        # Extract the prediction, probability, and actual values from the test set
        preds = self.model.transform(test_data).select("target", "prediction", 'probability')
        preds = preds.withColumn("probability_array", vector_to_array(col("probability")))
        preds = preds.withColumn("index_0_probability", col("probability_array")[0])
        preds = preds.withColumn("index_1_probability", col("probability_array")[1])
        preds = preds.withColumn("index", F.monotonically_increasing_id())
        
        # Align predictions with the original test set for further analysis
        original_test_set = original_test_set.withColumn("index", F.monotonically_increasing_id())
        plot_set = np.array(preds.join(original_test_set, on="index", how="outer").select("prediction", 'index_1_probability', 'price', 'timestamp').collect())
    

        # Convert predictions to numpy array for sklearn metrics
        preds_numpy = np.array(preds.select("target", "prediction", 'index_0_probability','index_1_probability').collect())
        y_true = preds_numpy[:,0]
        y_pred = preds_numpy[:,1]
        y_prob = preds_numpy[:,-2:]

        # Compute evaluation metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # Log metrics
        scores = {
            'train_roc': train_roc,
            'test_roc': test_roc,
            'accuracy': acc,
            'precision': precision,
            'recall':recall
        }
                
        wandb.log(scores)

        # Log additional wandb visualizations
        sklearn_model = self.model._xgb_sklearn_model
        wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels=['sell', 'buy'])
        wandb.sklearn.plot_roc(y_true, y_prob, labels=['sell', 'buy'])
        wandb.sklearn.plot_precision_recall(y_true, y_prob, labels=['sell', 'buy'])
        wandb.sklearn.plot_feature_importances(sklearn_model, feature_columns)

        # Log bokeh and plotly visualizations
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
        

    def save_model(self, path: str = 'models/xgboost_model'):
        """
        Saves the trained XGBoost model to a specified path.

        Args:
            path (str): The local path to save the model. Defaults to 'models/xgboost_model'.
        """
        if not self.model:
            raise ValueError("No model found. Train the model before saving.")
        
        current_timestamp = datetime.now()
        timestamp_str = current_timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        path = f"{path}_{timestamp_str}"
        
        # Save the model locally
        if not os.path.exists(path):
            os.makedirs(path)

        self.model.save(path)
        logging.info(f"Model saved successfully at {path}.")

if __name__=='__main__':
    try:
        btc = BTCDataloader()
        btc.setup_spark()
        df = btc.load_data()
        train_data, test_data, preproc_spark = btc.preproc_split(df)
        
        
        xg = XGBoostTrainer()
        xg.setup()
        
        
        model = xg.train_model(train_data=train_data, xgb_estimator=xg.model)
        xg.score_model(train_data=train_data, test_data=test_data, feature_columns=btc.feature_columns, original_test_set=btc.test_set)
        xg.save_model()
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)