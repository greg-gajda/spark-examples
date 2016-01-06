# Tuning Random Forest algorithm

The whole tuning program which tests Random Forest algorithm with different number of trees:

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
val featureSubsetStrategy = "auto"
val impurity = "entropy"
val maxDepth = 20
val maxBins = 34

val tuning = for (numTrees <- Range(2, 20)) yield {
  val model = RandomForest.trainClassifier(train, numClasses, BikeBuyerModel.categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
  val predictionsAndLabels = test.map {
    point => (model.predict(point.features), point.label)
  }
  val stats = Stats(confusionMatrix(predictionsAndLabels))
  val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
  (numTrees, stats.MCC, stats.ACC, metrics.areaUnderPR, metrics.areaUnderROC)
} 
tuning.sortBy(_._2).reverse.foreach{
  x => println(x._1 + " " + x._2 + " " + x._3+ " " + x._4+ " " + x._5)
}

sc.stop()
```

Sample output sorted by Matthews correlation coefficient:

<div class = "console">
15 0.7710216770066503 0.885483870967742 0.9149414393859618 0.8854838709677418<br>
12 0.7673416812762877 0.8833333333333333 0.9100277971969772 0.8833333333333333<br>
13 0.7655913978494624 0.8827956989247312 0.9120967741935484 0.8827956989247312<br>
14 0.7594366282306201 0.8795698924731182 0.9080105277365367 0.8795698924731183<br>
9 0.759352282307894 0.8795698924731182 0.9113187437828619 0.8795698924731182<br>
...
</div>