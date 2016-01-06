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
Using case class that reflects raw data can make conversion into LabeledPoints a bit easier:
```scala
case class HouseModel(id: Long,
                      date: String,
                      price: Float,
                      bedrooms: Int,
                      bathrooms: Float,
                      sqft_living: Int,
                      sqft_lot: Int,
                      floors: Float,
                      waterfront: Int,
                      view: Int,
                      condition: Int,
                      grade: Int,
                      sqft_above: Int,
                      sqft_basement: Int,
                      yr_built: Int,
                      yr_renovated: Int,
                      zipcode: String,
                      lat: Float,
                      long: Float,
                      sqft_living15: Int,
                      sqft_lot15: Int)
    extends LabeledPointConverter {

  def label() = price.toDouble
  def features() = HouseModel.convert(this)
}

```


Data is loaded following way: