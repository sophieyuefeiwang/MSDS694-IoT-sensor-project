from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from pyspark.sql import SparkSession
from pysparkling import H2OContext

from user_definition import *

ss = SparkSession.builder.config('spark.ext.h2o.log.level', 'FATAL').getOrCreate()
ss.sparkContext.setLogLevel('OFF')
hc = H2OContext.getOrCreate()

# step 1
# create spark dataframe
train_df = ss.read.parquet(train_folder).repartition(8).cache()
valid_df = ss.read.parquet(valid_folder).repartition(8).cache()

# convert spark dataframe to h2oFrame
train_h2o = hc.asH2OFrame(train_df, "train")
valid_h2o = hc.asH2OFrame(valid_df, "valid")

# convert label column to categorical datatype
train_h2o['label'] = train_h2o['label'].asfactor()
valid_h2o['label'] = valid_h2o['label'].asfactor()

for i in train_h2o.types:  # dict
   print(f"{i} - {train_h2o.types[i]}")
print('')


# step 2
predictors = train_h2o.names[:]
response = 'label'
predictors.remove(response)
for i in predictors:
   print(i)
print('')


# step 3
model_xg = H2OXGBoostEstimator(max_runtime_secs=max_runtime_secs,
                               nfolds=n_fold,
                               seed=seed)
model_xg.train(x=predictors,
               y=response,
               training_frame=train_h2o,
               validation_frame=valid_h2o
            )
print(round(model_xg.auc(valid=True),n_digits)) 
print('')
print(model_xg.confusion_matrix(valid=True))
print('')


# step 4
model_dl =H2ODeepLearningEstimator(nfolds=n_fold, 
                                   variable_importances=True,
                                   loss='Automatic',
                                   max_runtime_secs=max_runtime_secs,
                                   seed=seed)
model_dl.train(x=predictors,
               y=response,
               training_frame=train_h2o,
               validation_frame=valid_h2o)

print(round(model_dl.auc(valid=True),n_digits)) 
print('')
print(model_dl.confusion_matrix(valid=True)) 

ss.stop()


