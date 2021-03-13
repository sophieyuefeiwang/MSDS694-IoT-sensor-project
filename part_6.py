import h2o
from h2o.automl import H2OAutoML

from pyspark.sql import SparkSession
from pysparkling import *

from user_definition import *

ss = SparkSession.builder.config('spark.ext.h2o.log.level', 'FATAL')\
    .getOrCreate()
ss.sparkContext.setLogLevel('OFF')
hc = H2OContext.getOrCreate()

# mute the progress bar
h2o.no_progress()

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


# step 2
predictors = train_h2o.names[:]
response = 'label'
predictors.remove(response)

# step 3
model_automl = H2OAutoML(max_runtime_secs=max_runtime_secs,
                         seed=seed, nfolds=n_fold)
model_automl.train(x=predictors,
                   y=response,
                   training_frame=train_h2o,
                   validation_frame=valid_h2o)


print(model_automl.leaderboard)

print(round(model_automl.leader.auc(valid=True), n_digits))

print(model_automl.leader.confusion_matrix(valid=True))

ss.stop()

# spark-submit --executor-memory 12g
# --driver-memory 12g --executor-cores 6
