# Databricks notebook source
df = spark.read.format('csv').options(header='true', inferSchema='true').load('s3://lepuribigdata/Tweets.csv')

df= df[(df['text']!="null")]
display(df)

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover 
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import StringIndexer

tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="features")
indexer = StringIndexer(inputCol="airline_sentiment", outputCol="label")
pipeline_stages = [tokenizer, remover, hashingTF, indexer]
pipeline = Pipeline(stages=pipeline_stages)
pipeline_model = pipeline.fit(df)
processed_data = pipeline_model.transform(df)


display(processed_data)

# COMMAND ----------

train_data, test_data = processed_data.randomSplit([0.8, 0.2], seed = 3000)

numberFolds=3

lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=5)
paramGrid = (ParamGridBuilder()
            .addGrid(lr.regParam, [0.1, 0.5, 2.0, 4.0, 5.0])
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 0.7, 1.0])
            .addGrid(lr.maxIter, [1, 5, 10, 15])
            .build())
evaluator = MulticlassClassificationEvaluator(labelCol='label', metricName="accuracy")
crossValidator = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=numberFolds)
crossValidator_Model = crossValidator.fit(train_data)
predicted_data = crossValidator_Model.transform(test_data)

# COMMAND ----------

display(predicted_data)

# COMMAND ----------

#from pyspark.ml.evaluation import MulticlassClassificationEvaluator 
#evaluator = MulticlassClassificationEvaluator(metricName = "accuracy")
accuracy = evaluator.evaluate(predicted_data)
print(accuracy)

# COMMAND ----------

from pyspark.mllib.evaluation import MulticlassMetrics

# Compute raw scores on the test set
predictionAndLabels = predicted_data.map(lambda data: data.prediction, data.label))

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Overall statistics
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

# Statistics by class
labels = data.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

# Weighted stats
print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
