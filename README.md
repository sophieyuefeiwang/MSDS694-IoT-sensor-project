# MSDS694-IoT-sensor-project
Distributed Computing Project -- Activity Recognition using smartphone and smartwatch data in Apache Spark

# Project Description
The dataset used is from 'UCI WISDM Smartphone and Smartwatch Activity and Biometrics' which contains information collected by gyroscopes or accelerometers of smartphone and smartwatch. The goal is to classify and recognize human activity categories by applying machine learning techniques in a distributed computing setting (SparkML and Spark+H2O). 

**The project consists of six parts (including EDA and machine learning):**
  ## [Part 1](https://github.com/sophieyuefeiwang/MSDS694-IoT-sensor-project/blob/main/part_1.py)
  1) Load all data from subfolders at once as RDDs.
  2) Remove all the null values
  3) Convert RDDs to Spark dataframe
  4) Join the activity code dataframe with sensor info dataframe
  
  ## [Part 2] (https://github.com/sophieyuefeiwang/MSDS694-IoT-sensor-project/blob/main/part_2.py)
  1) Identify which activity is related to eating
  2) Check the number of activity types for each device, sensor and user
  3) Check the min, max, std, percentiles of the readings from gyroscopes or accelerometers
  
  ## [Part 3] (https://github.com/sophieyuefeiwang/MSDS694-IoT-sensor-project/blob/main/part_3.py)
  1) Encode the categorical column by first applying StringIndexer and then OneHotEncoder
  2) Combine all the feature columns using Vector Assembler 
  3) Scale the assembled features by StandardScaler
  4) Divide the dataset into training (80%) and test set(20%)
  5) Fit a logistic regression model with cross validation
  6) Check the evaluation metric areaUnderROC on the test set (**0.611**)

  ## [Part 4] (https://github.com/sophieyuefeiwang/MSDS694-IoT-sensor-project/blob/main/part_4.py)
  1) Fit a random forest classifier model with cross validation
  2) Check the evaluation metric areaUnderROC on the test set (**0.803**, much better than the logistic regression)
  3) Fit a gradient boosted tree classifier model with cross validation
  2) Check the evaluation metric areaUnderROC on the test set (**0.933**, better than the random forest classifier)
  
  ## [Part 5] (https://github.com/sophieyuefeiwang/MSDS694-IoT-sensor-project/blob/main/part_5.py)(note: part 3-4 uses SparkML, part 5-6 uses Sparkling Water--H2O with Spark)
  1) Fit a H2O gradient boosted tree classifier model with cross validation
  2) Check the evaluation metric areaUnderROC on the test set (**0.866**)
  3) Fit a H2O deep learning model with cross validation
  4) Check the evaluation metric areaUnderROC on the test set (**0.945**)
 
  ## [Part 6] (https://github.com/sophieyuefeiwang/MSDS694-IoT-sensor-project/blob/main/part_6.py)
  1) Apply AutoML on the dataset and return the leaderboard ![Screen Shot 2021-03-15 at 12 30 03 PM](https://user-images.githubusercontent.com/53668668/111210182-2c4a9b00-858a-11eb-8163-abbe71eeeae2.png)
  2) Fit the leader model from the screenshot above and check the evaluation metric areaUnderROC on the test set (**0.9597**, highest score so far!)

  
  
By comparing Spark ML and H2O, H2O is much easier to use since it takes care of all the data-preprocessing steps automatically (StringIndexer, OneHotEncoder, Vector Assembler, StandardScaler). The AutoML in H2O package is even simplier since it automatically search for the best performing algorithm and provide a lot of model interpretation visualizations (feature importance for example). Note: If you want to use H2O instead of Spark ML, you have to convert Spark dataframe (row-based) to H2O Frame (column-based). 
  
  
  
  
 
