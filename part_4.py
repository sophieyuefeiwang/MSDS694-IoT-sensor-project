from pyspark.sql import *
from pyspark.ml import *

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline

from builtins import round

from user_definition import *

ss = SparkSession.builder.getOrCreate()


# step 1
train_df = ss.read.parquet(train_folder).repartition(8).cache()
valid_df = ss.read.parquet(valid_folder).repartition(8).cache()

print(train_df.count())
print('')
print(valid_df.count())
print('')

# step 2
rf = RandomForestClassifier()
evaluator = BinaryClassificationEvaluator()

paraGrid = ParamGridBuilder().addGrid(rf.numTrees, num_trees).build()

cv = CrossValidator(estimator=rf,
                    evaluator=evaluator,
                    numFolds=n_fold,
                    estimatorParamMaps=paraGrid)

cvmodel = cv.fit(train_df)

rfpredicts = cvmodel.bestModel.transform(valid_df)

print('RandomForestClassifier')
print(cvmodel.bestModel.getNumTrees)
print(round(evaluator.evaluate(rfpredicts), n_digits))
print('')

# step 3
GBT = GBTClassifier()
evaluator = BinaryClassificationEvaluator()  # areaUnderROC is default

paramGrid = ParamGridBuilder().addGrid(GBT.maxDepth, max_depth).build()

cv = CrossValidator(estimator=GBT,
                    evaluator=evaluator,
                    numFolds=n_fold,
                    estimatorParamMaps=paramGrid)

cvmodel = cv.fit(train_df)

GBTpredicts = cvmodel.bestModel.transform(valid_df)

print('GBTClassifier')
print(cvmodel.bestModel.getMaxDepth())
print(round(evaluator.evaluate(GBTpredicts), n_digits))

ss.stop()

# spark-submit --executor-memory 12g 
# --driver-memory 12g --executor-cores 6
