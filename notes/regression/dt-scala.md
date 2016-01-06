# Decision tree in Scala

Function which creates decision tree model for regression can look like that:

```scala
  def createDecisionTreeRegressionModel(rdd: RDD[LabeledPoint]) = {
    val impurity = "variance"
    val maxDepth = 10
    val maxBins = 20
    DecisionTree.trainRegressor(rdd, Map[Int, Int](), impurity, maxDepth, maxBins)
  }
```