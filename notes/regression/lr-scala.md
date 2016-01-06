# Linear regression in Scala

Function which creates linear regression model can look like that:

```scala
  def createLinearRegressionModel(rdd: RDD[LabeledPoint]) = {
    LinearRegressionWithSGD.train(rdd, numIterations = 100, stepSize = 0.01)
  }
```