# Predicting House prices by using different regression algorithms

Data set used in this example can be found in data folder on github (https://github.com/grzegorzgajda/spark-examples) in file house-data.csv.

Evaluation linear regression algorithm, at least few things ought to be taken into account:
* Linear regression model 
* Model performance

Application name is set to:

```scala
val regressionApp = "Decision Tree Algorithm as classifier of Bike Buyers"
```
Spark's linear regression like classification algorithms work with LabeledPoint data structure.
Using case class that reflects raw data can make conversion into LabeledPoints a bit easier:
```scala
case class HouseModel(id: Long,
                      date: java.sql.Date,
                      price: Double,
                      bedrooms: Int,
                      bathrooms: Double,
                      sqft_living: Int,
                      sqft_lot: Int,
                      floors: Double,
                      waterfront: Int,
                      view: Int,
                      condition: Int,
                      grade: Int,
                      sqft_above: Int,
                      sqft_basement: Int,
                      yr_built: Int,
                      yr_renovated: Int,
                      zipcode: String,
                      lat: Double,
                      long: Double,
                      sqft_living15: Int,
                      sqft_lot15: Int)
    extends LabeledPointConverter {

  def label() = price.toDouble
  def features() = HouseModel.convert(this)
}
```
HouseModel companion object is overridden and together with apply method it provides method for conversion to Vector and marking categorical features
```scala
object HouseModel {

  def df = new java.text.SimpleDateFormat("yyyyMMdd'T'hhmmss")

  def apply(row: Array[String]) = new HouseModel(
    row(0).toLong, new java.sql.Date(df.parse(row(1)).getTime),
    row(2).toInt, row(3).toInt,
    row(4).toDouble, row(5).toInt, row(6).toInt,
    row(7).toDouble, row(8).toInt, row(9).toInt,
    row(10).toInt, row(11).toInt, row(12).toInt,
    row(13).toInt, row(14).toInt, row(15).toInt, row(16),
    row(17).toDouble, row(18).toDouble, row(19).toInt, row(20).toInt)

  def convert(model: HouseModel) = Vectors.dense(
    model.id.toDouble,
    model.bedrooms.toDouble,
    model.bathrooms,
    model.sqft_living.toDouble,
    model.sqft_lot.toDouble,
    model.floors,
    model.waterfront.toDouble,
    model.view.toDouble,
    model.condition.toDouble,
    model.grade.toDouble,
    model.sqft_above.toDouble,
    model.sqft_basement.toDouble,
    model.yr_built.toDouble,
    model.yr_renovated.toDouble,
    model.lat,
    model.long,
    model.sqft_living15.toDouble,
    model.sqft_lot15.toDouble)
}
```

Data is loaded following way:
```scala
val houses = hdFile.map(_.split(",")).
  filter { t => catching(classOf[NumberFormatException]).opt(t(0).toLong).isDefined }.
  map { HouseModel(_).toLabeledPoint() }
```
And splitted into training and test set:
```scala
val Array(train, test) = houses.randomSplit(Array(.9, .1), 10204L)
```
In regression it is recommended that the input variables have a mean of 0. It's easy to achieve by using the StandardScaler from Spark MLLib.

```scala
val scaler = new StandardScaler(withMean = true, withStd = true).fit(train.map(dp => dp.features))
val scaledTrain = train.map(dp => new LabeledPoint(dp.label, scaler.transform(dp.features))).cache()
val scaledTest = test.map(dp => new LabeledPoint(dp.label, scaler.transform(dp.features))).cache()
```
Summary statistics can be calculated:
```scala
val stats2 = Statistics.colStats(scaledTrain.map { x => x.features })
println(s"Max : ${stats2.max}, Min : ${stats2.min}, and Mean : ${stats2.mean} and Variance : ${stats2.variance}")
```
