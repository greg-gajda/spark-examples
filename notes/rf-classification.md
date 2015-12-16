A classification ensemble model is the combination of two or more classification models.
Random forest is an ensemble tree technique that builds a model with defined number of decision trees 
that is better classifier than average tree in the forest. So the whole performs better than any of its parts.

In Spark Random Forest model is created on decision tree algorithm with default voting strategy, 
where each tree votes on classification, and higher number of votes wins.

According to Spark documentation Random forests are one of the most successful machine learning models for classification and regression,
with reduced risk of over-fitting. 
The algorithm injects randomness into the training process so that each decision tree is a bit different.

Scala program:

```scala
val sc = new SparkContext(configLocalMode)
val bbFile = localFile(sc)

val data = bbFile.map { row =>
  BikeBuyerModel(row.split("\\t")).toLabeledPoint
}

val Array(train, test) = data.randomSplit(Array(.9, .1), 102059L)
train.cache()
test.cache()

val numClasses = 2
val numTrees = 10
val featureSubsetStrategy = "auto"
val impurity = "entropy"
val maxDepth = 20
val maxBins = 34

val model = RandomForest.trainClassifier(train, numClasses, BikeBuyerModel.categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

test.take(5).foreach {
  x => println(s"Predicted: ${model.predict(x.features)}, Label: ${x.label}")
}

val predictionsAndLabels = test.map {
  point => (model.predict(point.features), point.label)
}

val stats = Stats(confusionMatrix(predictionsAndLabels))
println(stats.toString)

val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
printMetrics(metrics)

sc.stop()
```
