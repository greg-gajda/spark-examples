# Model evaluation

Regardles of method of creating regression model, linear:
```scala
val model = createLinearRegressionModel(scaledTrain)
```
or based on decision tree:
```scala
val model = createDecisionTreeRegressionModel(scaledTrain)
```
evaluation requires of using test data to calculate predictions and store them with actual prices:
```scala
val predictionsAndValues = scaledTest.map {
  point => (model.predict(point.features), point.label)
}
```
Mean house price:
```scala
scaledTest.map { x => x.label }.mean()
```
Max prediction error:
```scala
predictionsAndValues.map { case (p, v) => math.abs(v - p) }.max
```
Root mean squared error:
```scala
math.sqrt(predictionsAndValues.map { case (p, v) => math.pow((v - p), 2) }.mean())
```
