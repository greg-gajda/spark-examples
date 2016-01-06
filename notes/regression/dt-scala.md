# Decision tree in Scala

Function which creates decision tree model for regression can look like that:

```scala
  def createDecisionTreeRegressionModel(rdd: RDD[LabeledPoint])(maxDepth:Int = 10, maxBins:Int = 20) = {
    val impurity = "variance"
    DecisionTree.trainRegressor(rdd, Map[Int, Int](), impurity, maxDepth, maxBins)
  }
```