from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from xgboost.spark import SparkXGBClassifier
from btc_streamer.ml.preprocessing import BTCDataloader
from btc_streamer.ml.model import XGBoostTrainer
import wandb

wandb.init(project="btc_crossval")


def cross_validator():
    
    btc = BTCDataloader()
    btc.setup_spark()
    df = btc.load_data()
    train_data, test_data, preproc_spark = btc.preproc_split(df)


    xgb_classifier = SparkXGBClassifier(
                features_col='features',
                label_col='target',
                num_workers=4,
                device='cpu',
            # booster='gbtree',
                eval_metric='logloss',
                validation_indicator_col = 'isVal',
            # eval_metric='auc',
            )

    xgb_classifier.setParams(early_stopping_rounds=5)

    paramGrid = ParamGridBuilder() \
        .addGrid(xgb_classifier.max_depth, [3, 5, 7]) \
        .addGrid(xgb_classifier.learning_rate, [0.1, 0.05, 0.01]) \
        .addGrid(xgb_classifier.n_estimators, [100, 200]) \
        .addGrid(xgb_classifier.subsample, [0.8, 1.0]) \
        .build()
        
    evaluator = BinaryClassificationEvaluator(
        labelCol='target',
        rawPredictionCol='rawPrediction',
        metricName='areaUnderROC'
    )

    crossval = CrossValidator(estimator=xgb_classifier,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=3)  

    cv_model = crossval.fit(train_data)

    best_model = cv_model.bestModel

    model = XGBoostTrainer().setup_spark()
    model.score_model(test_data, btc.feature_columns, btc.test_set, best_model)

    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    try:
        cross_validator()
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)