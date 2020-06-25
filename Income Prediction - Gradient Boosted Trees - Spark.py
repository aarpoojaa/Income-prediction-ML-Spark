# Databricks notebook source - 873406040 - POOJA RADHAKRISHNAN
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType
from pyspark.ml.feature import StringIndexer, Bucketizer
from pyspark.ml import Pipeline
import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

sc = spark.sparkContext

trainDataFile = "FileStore/tables/income_evaluation.csv"

df = spark.read.format("csv").option("header", True).option("inferSchema", True).option("ignoreLeadingWhiteSpace", True).option("mode", "dropMalformed").load(trainDataFile)
df = df.select(['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'income'])
print("Raw data:", df.count())

df = df.filter(f.col('workclass') != '?')
df = df.filter(f.col('occupation') != '?')
print("Valid data:", df.count())
df.show()

# COMMAND ----------

#DATA PREPROCSSSING:
splits = [-float("inf"), 30, 40, 50, 60, float("inf")]
tempBucketizer = Bucketizer(splits=splits, inputCol="age", outputCol="AgeBucket")
workClassIndexer = StringIndexer(inputCol='workclass', outputCol='workClassIndex')
maritalIndexer = StringIndexer(inputCol='marital-status', outputCol='maritalIndex')
occupationIndexer = StringIndexer(inputCol='occupation', outputCol='occupationIndex')
relationshipIndexer = StringIndexer(inputCol='relationship', outputCol='relationshipIndex')
raceIndexer = StringIndexer(inputCol='race', outputCol='raceIndex')
sexIndexer = StringIndexer(inputCol='sex', outputCol='sexIndex')
splits_hours = [-float("inf"), 20, 40, 60, float("inf")]
tempBucketizer_hours = Bucketizer(splits=splits_hours, inputCol="hours-per-week", outputCol="hoursBucket")
incomeIndexer = StringIndexer(inputCol='income', outputCol='incomeIndex')

train, test = df.randomSplit([0.7, 0.3], seed = 2020)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

#GRADIENT-BOOSTED TREE CLASSIFIER

from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10)

myStages4=[tempBucketizer, workClassIndexer, maritalIndexer, occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, tempBucketizer_hours, incomeIndexerLabel, vecAssem, gbt]
p4 = Pipeline(stages=myStages4)
p4Model = p4.fit(train)
predictions4 = p4Model.transform(test)
#predictions.show()
predictions4.select('label', 'features', 'probability', 'prediction').show(5)

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions4, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

testFiles = test.repartition(50)
dbutils.fs.rm("FileStore/tables/Income/", True)
dbutils.fs.rm("FileStore/tables/Fifa/", True)
dbutils.fs.rm("FileStore/tables/routes.csv", True)
dbutils.fs.rm("FileStore/tables/FIFA.csv", True)
dbutils.fs.rm("FileStore/tables/airports.csv", True)
dbutils.fs.rm("FileStore/tables/heartTesting.csv", True)
dbutils.fs.rm("FileStore/tables/heartTraining.csv", True)
testFiles.write.format("csv").option("header", True).save("FileStore/tables/Income/")
sourceStream = spark.readStream.format("csv").option("header", True).option("maxFilesPerTrigger", 1).load("dbfs:///FileStore/tables/Income")
incomeWindow = sourceStream.withWatermark("time", "60 seconds").groupBy(f.window("time", "30 seconds", "10 seconds"), 'income')
sinkStream = incomeWindow.writeStream.outputMode("complete").format("memory").queryName("incomeWindowQuery").start()
current = spark.sql("SELECT * FROM incomeWindowQuery")
current.orderBy("window","count").show(20, False)

# COMMAND ----------

#RANDOM FOREST CLASSIFIER

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features")

myStages2=[tempBucketizer, workClassIndexer, maritalIndexer, occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, tempBucketizer_hours, incomeIndexerLabel, vecAssem, rf]
p2 = Pipeline(stages=myStages2)
p2Model = p2.fit(train)
predictions2 = p2Model.transform(test)
#predictions.show()
predictions2.select('label', 'features', 'probability', 'prediction').show(5)

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions2, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

#LOGISTIC REGRESSION:

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=5, regParam=0.005)
incomeIndexerLabel = StringIndexer(inputCol='income', outputCol='label')


vecAssem = VectorAssembler(inputCols=['AgeBucket', 'workClassIndex', 'education-num', 'maritalIndex', 'occupationIndex', 'relationshipIndex', 'raceIndex', 'sexIndex', 'hoursBucket'], outputCol='features')
myStages=[tempBucketizer, workClassIndexer, maritalIndexer, occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, tempBucketizer_hours, incomeIndexerLabel, vecAssem, lr]
p = Pipeline(stages=myStages)
pModel = p.fit(train)
predictions = pModel.transform(test)
#predictions.show()
predictions.select('label', 'features', 'probability', 'prediction').show(5)

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

#DECISION TREE CLASSIFIER

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 10)

myStages3=[tempBucketizer, workClassIndexer, maritalIndexer, occupationIndexer, relationshipIndexer, raceIndexer, sexIndexer, tempBucketizer_hours, incomeIndexerLabel, vecAssem, dt]
p3 = Pipeline(stages=myStages3)
p3Model = p3.fit(train)
predictions3 = p3Model.transform(test)
#predictions.show()
predictions3.select('label', 'features', 'probability', 'prediction').show(5)


evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions3, {evaluator.metricName: "areaUnderROC"})))
