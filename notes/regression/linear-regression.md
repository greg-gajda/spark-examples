# Predicting House prices by using different linear regression algorithms

Data set used in this example can be found in data folder on github (https://github.com/grzegorzgajda/spark-examples) in file house-data.csv.

Evaluation linear regression algorithm, at least few things ought to be taken into account:
* Linear regression model 
* Model performance

Application name is set to:

```scala
val classificationApp = "Classification of customers by using Decision Tree"
```
Spark's linear regression like classification algorithms work with LabeledPoint data structure.



Data is loaded following way: