from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import asc, desc

from user_definition import *

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext  # Have to create sparckContext in order to create RDD

data = sc.wholeTextFiles(files)

# step 1
num_files = data.map(lambda x: x[0]).distinct().count()
print(num_files)
print('')

# step 2
cleaned_rdd = data.map(lambda x: (x[0], x[1].split('\n')))\
    .flatMapValues(lambda x: x)\
    .map(lambda x: (x[0].split('/'), x[1]))\
    .map(lambda x: (x[0][-2], x[0][-3], x[1][:-1].split(',')))\
    .filter(lambda x: all(x[2]))\
    .map(lambda x: (int(x[2][0]), x[0], x[1], x[2][1], int(x[2][2]),
                    float(x[2][3]), float(x[2][4]), float(x[2][5])))

num_records_no_null = cleaned_rdd.count()

print(num_records_no_null)
print('')

# step 3
schema = StructType([StructField('subject_id', IntegerType(), False),
                     StructField('sensor', StringType(), False),
                     StructField('device', StringType(), False),
                     StructField('activity_code', StringType(), False),
                     StructField('timestamp', LongType(), False),
                     StructField('x', FloatType(), False),
                     StructField('y', FloatType(), False),
                     StructField('z', FloatType(), False)])

data_df = ss.createDataFrame(cleaned_rdd, schema)
data_df.printSchema()

# step 4
unique_id_df = data_df.select('subject_id').distinct().sort(
    'subject_id', ascending=True)
unique_id_df.show(unique_id_df.count())
print('')

# step 5
unique_sensor_df = data_df.select(
    'sensor').distinct().sort('sensor', ascending=True)
unique_sensor_df.show(unique_sensor_df.count())
print('')

# step 6
unique_act_code_df = data_df.select(
    'activity_code').distinct().sort('activity_code', ascending=True)
unique_act_code_df.show(unique_act_code_df.count())

# step 7
result7 = data_df.filter(f"subject_id == '{subject_id}' and\
    activity_code == '{activity_code}'")\
    .orderBy(['timestamp', 'sensor'], ascending=[True, False])
result7.show(n)

# step 8
result8 = data_df.filter(f"subject_id == '{subject_id}'\
    and activity_code == '{activity_code}'")\
    .orderBy(['timestamp', 'sensor'], ascending=[True, False])

result8 = result8.withColumn('x_positive', result8['x'] >= 0)\
    .withColumn('y_positive', result8['y'] >= 0)\
    .withColumn('z_positive', result8['z'] >= 0)\
    .drop('x', 'y', 'z')

result8.show(n)

ss.stop()
