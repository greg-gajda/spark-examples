# Linear regression in Scala

Function which creates linear regression model can look like that:

```scala
  def createLinearRegressionModel(rdd: RDD[LabeledPoint], numIterations: Int = 100, stepSize: Double = 0.01) = {
    LinearRegressionWithSGD.train(rdd, numIterations, stepSize)
  }
```