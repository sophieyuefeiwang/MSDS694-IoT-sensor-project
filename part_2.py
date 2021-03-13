from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *

from user_definition import *

# step 1
ss = SparkSession.builder.config("spark.executor.memory", "5g")\
                         .config("spark.driver.memory", "5g").getOrCreate()

# .config('spark.driver.extraClassPath','postgresql-42.2.18.jar')

activity_code = ss.read.jdbc(url=url, table=table, properties=properties)

num_distinct_act = activity_code.distinct().count()
print(num_distinct_act)
print('')


# step 2
activity_code.orderBy('activity', ascending=False).show(
    truncate=False)  # Show the full name by using truncate


# step 3
def check_eating(x):
    tracker = 0
    for i in eating_strings:
        if i in x:
            tracker = tracker + 1
    if tracker >= 1:
        return True
    else:
        return False


# Register the function as UDF
check_eating_udf = udf(check_eating, BooleanType())   # From class example 3

eating_df = activity_code.withColumn('eating', check_eating_udf(lower(
                                     activity_code['activity']))).orderBy(
                                         'eating', 'code', ascending=[
                                             False, True])

eating_df.printSchema()

eating_df.show()


# step 4
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
files_df = create_activity_df(ss, files_rdd, schema)

result4 = files_df.select('subject_id', 'sensor', 'device', 'activity_code')\
                  .distinct().groupBy('subject_id', 'sensor', 'device')\
                  .count().orderBy('subject_id', 'device', 'sensor')

result4.show(result4.count())


# step 5
joined_df = files_df.join(
    activity_code, files_df.activity_code == activity_code.code).cache()

selected_joined_df = joined_df.select(
    'subject_id', 'activity', 'device',
    'sensor', 'x', 'y', 'z', 'activity_code').cache()

selected_joined_df.groupBy('subject_id', 'activity', 'device', 'sensor').\
                    agg(min('x').alias('x_min'),
                        min('y').alias('y_min'),
                        min('z').alias('z_min'),
                        avg('x').alias('x_avg'),
                        avg('y').alias('y_avg'),
                        avg('z').alias('z_avg'),
                        max('x').alias('x_max'),
                        max('y').alias('y_max'),
                        max('z').alias('z_max'),
                        expr('percentile(x, array(0.05))')[0].alias('x_05%'),
                        expr('percentile(y, array(0.05))')[0].alias('y_05%'),
                        expr('percentile(z, array(0.05))')[0].alias('z_05%'),
                        expr('percentile(x, array(0.25))')[0].alias('x_25%'),
                        expr('percentile(y, array(0.25))')[0].alias('y_25%'),
                        expr('percentile(z, array(0.25))')[0].alias('z_25%'),
                        expr('percentile(x, array(0.50))')[0].alias('x_50%'),
                        expr('percentile(y, array(0.50))')[0].alias('y_50%'),
                        expr('percentile(z, array(0.50))')[0].alias('z_50%'),
                        expr('percentile(x, array(0.75))')[0].alias('x_75%'),
                        expr('percentile(y, array(0.75))')[0].alias('y_75%'),
                        expr('percentile(z, array(0.75))')[0].alias('z_75%'),
                        expr('percentile(x, array(0.95))')[0].alias('x_95%'),
                        expr('percentile(y, array(0.95))')[0].alias('y_95%'),
                        expr('percentile(z, array(0.95))')[0].alias('z_95%'),
                        stddev('x').alias('x_std'),
                        stddev('y').alias('y_std'),
                        stddev('z').alias('z_std'))\
                    .orderBy('activity', 'subject_id', 'device', 'sensor')\
                    .show(n)


# step 6
extracted_joined = joined_df.select(
                    'subject_id', 'activity',
                    'timestamp', 'device', 'sensor', 'x', 'y', 'z').cache()

extracted_joined.filter(f"subject_id=={subject_id}")\
                .orderBy('timestamp', 'device', 'sensor')\
                .filter(lower(extracted_joined['activity'])
                        .contains(f"{activity_string}"))\
                .drop('subject_id').show(n)


# step 7
# Filer for rows that has both sensor
new_joined_df = joined_df.filter(f"subject_id=={subject_id}")\
                         .filter(lower(extracted_joined['activity'])
                                 .contains(f"{activity_string}"))

# Filer for rows that has both sensor
both_sensor_df = new_joined_df.groupBy('activity_code', 'device', 'timestamp')\
    .agg(countDistinct('sensor').alias('sensor_count'))\
    .filter('sensor_count == 2')

extracted_df = joined_df.drop('activity', 'code')

big_joined_df = extracted_df.join(
      both_sensor_df, ['activity_code', 'device', 'timestamp'], 'leftsemi')

accel = big_joined_df.filter("sensor == 'accel'")\
    .filter(f"subject_id == {subject_id}")\
    .withColumnRenamed('x', 'accel_x')\
    .withColumnRenamed('y', 'accel_y')\
    .withColumnRenamed('z', 'accel_z')


gyro = big_joined_df.filter("sensor == 'gyro'")\
    .filter(f"subject_id == {subject_id}")\
    .withColumnRenamed('x', 'gyro_x')\
    .withColumnRenamed('y', 'gyro_y')\
    .withColumnRenamed('z', 'gyro_z')

accel.join(gyro, ['activity_code', 'device', 'timestamp'])\
     .orderBy('activity_code', 'timestamp')\
     .drop('subject_id', 'sensor')\
     .show(n)

ss.stop()
