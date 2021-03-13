from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder


from user_definition import *

ss = SparkSession.builder.config("spark.executor.memory", "5g")\
                         .config("spark.driver.memory", "5g").getOrCreate()

# step 1
activity_code = ss.read.jdbc(
    url=url, table=table, properties=properties).coalesce(8).cache()

schema = StructType([StructField('subject_id', IntegerType(), False),
                     StructField('sensor', StringType(), False),
                     StructField('device', StringType(), False),
                     StructField('activity_code', StringType(), False),
                     StructField('timestamp', LongType(), False),
                     StructField('x', FloatType(), False),
                     StructField('y', FloatType(), False),
                     StructField('z', FloatType(), False)])

# Load the data to rdds
files_rdd = file_rdd(ss, files)
# Create the spark dataframe
files_df = create_activity_df(ss, files_rdd, schema).coalesce(8).cache()


# step 2
def check_eating(x):
    tracker = 0
    for i in eating_strings:
        if i in x:
            tracker = tracker + 1
    if tracker >= 1:
        return True
    else:
        return False


check_eating_udf = udf(check_eating, BooleanType())

eating_df = activity_code.withColumn(
    'eating', check_eating_udf(lower(activity_code['activity'])))

result2 = eating_df.filter('eating').select('code').distinct().sort('code')
result2.show()


# step 3
eating_df = eating_df.select(
    ['activity', 'code', col('eating').cast("integer")]).orderBy([])

joined_df = eating_df.join(files_df, eating_df.code ==
                           files_df.activity_code).cache()

result3 = joined_df.select('subject_id', 'sensor', 'device', 'activity_code',
                           'timestamp', 'x', 'y', 'z', 'eating')\
            .orderBy(['subject_id', 'timestamp', 'device', 'sensor']).cache()

result3.show(n)


# step 4
both_sensor_df = joined_df.groupBy('activity_code', 'device', 'timestamp')\
    .agg(countDistinct('sensor').alias('sensor_count'))\
    .filter('sensor_count==2').cache()

# join by the combination of three columns
result4_joined_df = joined_df.join(both_sensor_df, [
            'activity_code', 'device', 'timestamp'], 'leftsemi')\
    .select('sensor', 'activity', 'activity_code', 'subject_id',
            'device', 'timestamp', 'x', 'y', 'z', 'eating').distinct().cache()

accel = result4_joined_df.filter("sensor == 'accel'")\
    .withColumnRenamed('x', 'accel_x')\
    .withColumnRenamed('y', 'accel_y')\
    .withColumnRenamed('z', 'accel_z')

gyro = result4_joined_df.filter("sensor == 'gyro'")\
    .withColumnRenamed('x', 'gyro_x')\
    .withColumnRenamed('y', 'gyro_y')\
    .withColumnRenamed('z', 'gyro_z')

result4_df = accel.join(gyro, ['activity', 'device', 'timestamp'])\
    .select(gyro.activity_code, accel.subject_id,
            'timestamp', 'device', accel.eating,
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z').cache()

result4_count = result4_df.count()
print(result4_count)
print('')


# step 5
result5_df = result4_df

for i in range(1, window_size+1):
    result5_df = result5_df.withColumn(f"lead_{i}_accel_x",
                                       lead('accel_x', i)
                                       .over(Window.partitionBy([
                                        'subject_id',
                                        'activity_code',
                                        'device'])
                                        .orderBy(['subject_id',
                                                  'activity_code',
                                                  'device',
                                                  'timestamp'])))
    result5_df = result5_df.withColumn(f"lead_{i}_accel_y",
                                       lead('accel_y', i)
                                       .over(Window.partitionBy([
                                           'subject_id',
                                           'activity_code',
                                           'device'])
                                            .orderBy(['subject_id',
                                                      'activity_code',
                                                      'device',
                                                      'timestamp'])))
    result5_df = result5_df.withColumn(f"lead_{i}_accel_z",
                                       lead('accel_z', i)
                                       .over(Window.partitionBy([
                                           'subject_id',
                                           'activity_code',
                                           'device'])
                                            .orderBy(['subject_id',
                                                      'activity_code',
                                                      'device',
                                                      'timestamp'])))
    result5_df = result5_df.withColumn(f"lead_{i}_gyro_x",
                                       lead('gyro_x', i)
                                       .over(Window.partitionBy([
                                           'subject_id',
                                           'activity_code',
                                           'device'])
                                            .orderBy(['subject_id',
                                                      'activity_code',
                                                      'device',
                                                      'timestamp'])))
    result5_df = result5_df.withColumn(f"lead_{i}_gyro_y",
                                       lead('gyro_y', i)
                                       .over(Window.partitionBy([
                                           'subject_id',
                                           'activity_code',
                                           'device'])
                                            .orderBy(['subject_id',
                                                      'activity_code',
                                                      'device',
                                                      'timestamp'])))
    result5_df = result5_df.withColumn(f"lead_{i}_gyro_z",
                                       lead('gyro_z', i)
                                       .over(Window.partitionBy([
                                           'subject_id',
                                           'activity_code',
                                           'device'])
                                            .orderBy(['subject_id',
                                                      'activity_code',
                                                      'device',
                                                      'timestamp'])))

result5_df_new = result5_df.orderBy(
                        ['subject_id', 'activity_code',
                         'device', 'timestamp']).drop('activity_code').cache()

result5_df_new.show(n)


# step 6
result6_df = result5_df.orderBy(
    ['subject_id', 'activity_code', 'device', 'timestamp']).cache()


def indexStringColumns(df, cols):
    newdf = df
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c+'-num')
        sm = si.fit(newdf)
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+'-num', c)
    return newdf


result6_df_numeric = indexStringColumns(result6_df, ['device'])


def oneHotEncodeColumns(df, cols):
    newdf = df
    for c in cols:
        ohe = OneHotEncoder(inputCol=c, outputCol=c+'-onehot', dropLast=False)
        ohe_model = ohe.fit(newdf)
        newdf = ohe_model.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+'-onehot', c)
    return newdf


result6_df_onehot = oneHotEncodeColumns(result6_df_numeric, ['device'])\
                        .orderBy(['subject_id', 'timestamp', 'device'])

# Rearrange the order of the columns
cols = result6_df_onehot.columns  # this is a list of columns
sorted_cols = cols[:3]
sorted_cols.append(cols[-1])
sorted_cols.extend(cols[3:-1])
result6_df_new = result6_df_onehot.select(sorted_cols).cache()
result6_df_onehot = result6_df_new.drop('activity_code', 'eating').cache()
result6_df_onehot.show(n)


# step 7
result7_df = result6_df_new

cols = result7_df.columns

input_cols = cols[5:]

va = VectorAssembler(outputCol='features',
                     inputCols=input_cols, handleInvalid="skip")

result7_transformed = va.transform(result7_df).select(
    'activity_code', 'subject_id', 'timestamp', 'eating', 'device', 'features')


def standard_scaler(input_df):
    df = input_df

    scaler = StandardScaler(
        inputCol='features', outputCol='features_Scaled',
        withMean=True, withStd=True)

    stds = scaler.fit(df)

    # Normalize each feature
    df = stds.transform(df).drop('features')
    df = df.withColumnRenamed('features_Scaled', 'features')
    return df


result7_standard = standard_scaler(result7_transformed).cache()

result7_final = result7_standard.select('eating', 'device', 'features')\
            .orderBy(['subject_id', 'activity_code', 'device', 'timestamp'])

result7_final.show(n)


# step 8
result8_df = result7_final

input_cols_8 = ['features', 'device']
va8 = VectorAssembler(outputCol='features_new',
                      inputCols=input_cols_8, handleInvalid="skip")

result8_transformed = va8.transform(result8_df)\
                         .drop('features', 'device')\
                         .withColumnRenamed('features_new', 'features')\
                         .withColumnRenamed('eating', 'label')\
                         .select('features', 'label')

#result8_transformed.show(5)


# step 9
result9_df = result8_transformed

splits = result9_df.randomSplit([0.8, 0.2], seed=1)

train = splits[0].cache()
valid = splits[1].cache()

train.show(n)
valid.show(n)


# step 10
lr = LogisticRegression(regParam=0.01, maxIter=100, fitIntercept=True)

bceval = BinaryClassificationEvaluator()
cv = CrossValidator().setEstimator(lr).setEvaluator(bceval).setNumFolds(n_fold)

paramGrid = ParamGridBuilder().addGrid(lr.maxIter, max_iter)\
    .addGrid(lr.regParam, reg_params).build()

cv.setEstimatorParamMaps(paramGrid)

cvmodel = cv.fit(train)

print(cvmodel.bestModel.coefficients)
print('')
print(cvmodel.bestModel.intercept)
print('')
print(cvmodel.bestModel.getMaxIter())
print('')
print(cvmodel.bestModel.getRegParam())
print('')


# step 11
result11 = bceval.setMetricName('areaUnderROC').evaluate(
    cvmodel.bestModel.transform(valid))
print(result11)

ss.stop()
