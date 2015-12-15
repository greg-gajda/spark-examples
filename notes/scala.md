In Spark program first thing is to create SparkConf object that contains information about application, which is required by SparkContext object as constructor parameter. It is important that only one SparkContext may be active per JVM, and should be stopped before creating new context or before exiting current application. Application name is set to:

```
val applicationName = "Decision Tree Algorithm as classifier of Bike Buyers"
```


Function which returns configuration for running application in local mode with as many worker threads as logical cores available and with defined access to Cassandra is defined:

```
def local: SparkConf = {
    val conf = new SparkConf().setAppName(applicationName)
    conf.setMaster("local[*]")
    conf.set("spark.cassandra.connection.host", cassandraHost)
    conf
}
```
And to run application in Standalone Cluster mode:
```
def cluster: SparkConf = {
    val conf = new SparkConf().setAppName(applicationName)
    conf.setMaster("spark://192.168.1.15:7077")
    conf.setJars(Array("build/libs/spark-examples-1.0.jar"))
    conf.set("spark.cassandra.connection.host", cassandraHost)
    conf
}
```
Single node Cassandra is available on default port 9042 at:

  ```
  val cassandraHost = "127.0.0.1"
  ```

and to use it from Spark, location of Cassandra must be added to spark configuration as shown above.

Functions to load data from local storage, HDFS and Cassandra go as follow. To run on cluster, local file should be located on network storage available for every Spark’s workers or copied to exactly the same location on every node.
```
def localFile: (SparkContext => RDD[String]) = sc => {
    sc.textFile("data/bike-buyers")
}
```
Using HDFS requires Hadoop being configured and available. The only difference in code is that instead of providing file path, HDFS URL is to be supplied. 192.168.1.15:9000 reflects my local network Hadoop Cluster configuration, so it ought to be replaced with some alternative.
```
def hdfsFile: (SparkContext => RDD[String]) = sc => {
    sc.textFile("hdfs://192.168.1.15:9000/spark/bike-buyers")
}
```
CassandraRows are mapped into Strings, only to keep the same form, as after reading from text file. More reasonably solution could transform rows directly into something more useful.
```
def cassandraFile: (SparkContext => RDD[String]) = sc => {
    import com.datastax.spark.connector._
    sc.cassandraTable("spark", "bike_buyers").map { row => row.columnValues.mkString("\t") }
}
```
To load data into Cassandra simple ETL program written in Scala can look like this:
```
object LoadBikeBuyers {

  def main(args: Array[String]): Unit = {

    org.apache.log4j.BasicConfigurator.configure()
    val host = args(0) 
    val cc = com.datastax.spark.connector.cql.CassandraConnector(Set(InetAddress.getByName(host)))

    val keyspaceCql = Source.fromInputStream(getClass.getResourceAsStream("/create_spark_keyspace.cql")).mkString
    val tableCql = Source.fromInputStream(getClass.getResourceAsStream("/create_bike_buyers_table.cql")).mkString

    val bbFile = Source.fromFile("data/bike-buyers", "utf8").getLines()

    cc.withSessionDo(s => {
      s.execute(keyspaceCql)
      s.execute(tableCql)

      val columns = s.getCluster.getMetadata.getKeyspace("spark").getTable("bike_buyers").getColumns

      bbFile.map { x => x.split("\\t") }.foreach(row => {

        val insert = new QueryBuilder(s.getCluster).insertInto("spark", "bike_buyers")
        row.zipWithIndex.foreach(vi => {
          val column = columns(vi._2)
          insert.value(column.getName,
            if (column.getType == DataType.text()) vi._1
            else if (column.getType == DataType.cfloat()) vi._1.replaceFirst(",", ".").toFloat
            else vi._1.toInt)
        })
        s.execute(insert)
      })
    })
  }
}
```
Both scripts (create_spark_keyspace.cql, create_bike_buyers_table.cql) are on github. 
After executing of LoadBikeBuyers.scala, keyspace “spark” and table “bike_buyers” are created, and content of bike-buyers file is loaded into it. 
After loading data into RDD of Strings, conversion into LabeledPoint data structure can be prepared. For binary classification, labels should be negative or positive, represented by 0 or 1. Categorical features ought to be converted to numeric values 0, 1, 2 and so on. In this case, BikeBuyer flag would serve as Label, and all the rest would compose features vector. Customer Key doesn’t play any real decision role but helps prevent model overfitting. 
Using case class that reflects raw data can make conversion into LabeledPoints a bit easier:
```
case class BikeBuyerModel(customerKey: Int, age: Int, bikeBuyer: Int, commuteDistance: String, englishEducation: String, gender: String, houseOwnerFlag: Int, maritalStatus: String, numberCarsOwned: Int, numberChildrenAtHome: Int, englishOccupation: String, region: String, totalChildren: Int, yearlyIncome: Float)
    extends LabeledPointConverter {

  def label() = bikeBuyer.toDouble
  def features() = BikeBuyerModel.convert(this)

}
```
LabeledPointConverter is trait that could be reused. Case class build with this trait must provide implementation of label and feature.
```
trait LabeledPointConverter {
  def label(): Double
  def features(): Vector
  def toLabeledPoint() = LabeledPoint(label(), features())
}
```
BikeBuyerModel companion object is overridden and together with apply method it provides method for conversion to Vector and marking categorical features.
```
object BikeBuyerModel {

  def apply(row: Array[String]) = new BikeBuyerModel(
    row(0).toInt, row(1).toInt, row(2).toInt, row(3),
    row(4), row(5), row(6).toInt,
    row(7), row(8).toInt, row(9).toInt,
    row(10), row(11), row(12).toInt, row(13).replaceFirst(",", ".").toFloat)

  def categoricalFeaturesInfo() = {
    Map[Int, Int](2 -> 5, 3 -> 5, 4 -> 2, 6 -> 2, 9 -> 5, 10 -> 3)
  }

  def convert(model: BikeBuyerModel) = Vectors.dense(
    model.customerKey.toDouble,
    model.age.toDouble,
    model.commuteDistance match {
      case "0-1 Miles"  => 0d
      case "1-2 Miles"  => 1d
      case "2-5 Miles"  => 2d
      case "5-10 Miles" => 3d
      case "10+ Miles"  => 4d
    },
    model.englishEducation match {
      case "High School"         => 0d
      case "Partial High School" => 1d
      case "Partial College"     => 2d
      case "Graduate Degree"     => 3d
      case "Bachelors"           => 4d
    },
    model.gender match {
      case "M" => 0d
      case "F" => 1d
    },
    model.houseOwnerFlag.toDouble,
    model.maritalStatus match {
      case "S" => 0d
      case "M" => 1d
    },
    model.numberCarsOwned.toDouble,
    model.numberChildrenAtHome.toDouble,
    model.englishOccupation match {
      case "Professional"   => 0d
      case "Clerical"       => 1d
      case "Manual"         => 2d
      case "Management"     => 3d
      case "Skilled Manual" => 4d
    },
    model.region match {
      case "North America" => 0d
      case "Pacific"       => 1d
      case "Europe"        => 2d
    },
    model.totalChildren.toDouble,
    model.yearlyIncome)
}
```
Now, acquiring data in format required by Spark is quite easy:
```
val data = bbFile.map { row => BikeBuyerModel(row.split("\\t")).toLabeledPoint }
```
After that, data can be split into train and test parts, to conform cross-validation method, when model is trained with part of dataset and its performance is evaluated with another part:
```
val Array(train, test) = data.randomSplit(Array(.9, .1))
```
It seems to be a good moment to cache data for further reuse:
```
train.cache()
test.cache()
```
Now Spark’s classification decision tree algorithm can be trained:
```
val numClasses = 2
val impurity = "entropy" 
val maxDepth = 20
val maxBins = 24

val dtree = DecisionTree.trainClassifier(train, numClasses, BikeBuyerModel.categoricalFeaturesInfo(), impurity, maxDepth, maxBins)
```
Trained model can be used for prediction of whether potential customer is going to buy a bicycle or not:
```
    test.take(5).foreach {
      x => println(s"Predicted: ${dtree.predict(x.features)}, actual value: ${x.label}")
    }
```
Typical response of prediction for 5 top records from test dataset can look like this: 

Predicted: 1.0, Label: 1.0
Predicted: 1.0, Label: 1.0
Predicted: 0.0, Label: 0.0
Predicted: 0.0, Label: 1.0
Predicted: 1.0, Label: 1.0

To answer questions what is the real performance of this model, what is its ability to provide correct responses, some metrics has to be evaluated.
To further check performance of created model, test data can be used to collect predictions and expected values:
    val predictionsAndLabels = test.map { 
      point => (dtree.predict(point.features), point.label) 
    } 
In general, predictions of each data point from bike buyers dataset can be assigned to one of four categories: 
•	True Positive, when buyer is predicted as buyer,
•	True Negative, when not buyer is predicted as not buyer,
•	False Positive, when not buyer is predicted as buyer,
•	False Negative, when buyer predicted as not buyer.
Calculations can look like that:
    val (tp, tn, fp, fn) = predictionsAndLabels.aggregate((0, 0, 0, 0))(
      seqOp = (t, pal) => {
        val (tp, tn, fp, fn) = t
        (if (pal._1 == pal._2 && pal._2 == 1.0) tp + 1 else tp,
         if (pal._1 == pal._2 && pal._2 == 0.0) tn + 1 else tn,
         if (pal._1 == 1.0 && pal._2 == 0.0) fp + 1 else fp,
         if (pal._1 == 0.0 && pal._2 == 1.0) fn + 1 else fn)
      },
      combOp = (t1, t2) => (t1._1 + t2._1, t1._2 + t2._2, t1._3 + t2._3, t1._4 + t2._4))

Based on above, some other measures can be entered, e.g. in form of utility class:
class Stats(val tp: Int, val tn: Int, val fp: Int, val fn: Int) {

  val TPR = tp / (tp + fn).toDouble
  val recall = TPR
  val sensitivity = TPR
  val TNR = tn / (tn + fp).toDouble
  val specificity = TNR
  val PPV = tp / (tp + fp).toDouble
  val precision = PPV
  val NPV = tn / (tn + fn).toDouble
  val FPR = 1.0 - specificity
  val FNR = 1.0 - recall
  val FDR = 1.0 - precision
  val ACC = (tp + tn) / (tp + fp + fn + tn).toDouble
  val accuracy = ACC
  val F1 = 2 * PPV * TPR / (PPV + TPR).toDouble
  val MCC = (tp * tn - fp * fn).toDouble / math.sqrt((tp + fp).toDouble * (tp + fn).toDouble * (fp + tn).toDouble * (tn + fn).toDouble)

}
True positive rate TPR, called also recall or sensitivity is defined as the number of samples correctly predicted as belonging to the positive class (true positives) divided by the total number of elements that actually belong to the positive class (i.e. the sum of true positives and false negatives, which are items which were not predicted as belonging to the positive class but should have been)
True negative rate TNR, called also specificity is measure of samples correctly predicted as belonging to negative class divided by the total number of elements actually belonging to the negative class.
Positive predictive value, called also precision is the proportion of samples correctly predicted as belonging to the positive class (true positives) divided by the total number of elements predicted as belonging to the positive class (i.e. the sum of true positives and false positives, which are items incorrectly predicted as belonging to the class).
Negative predictive value is the proportion of samples correctly predicted as negative divided by the total number of true negative results.
False positive rate, called also fall-out is closely related to specificity and is equal to 1 - specificity.
False negative rate is closely related to sensitivity and is equal to 1 - recall.
False discovery rate is closely related to precision and is equal to 1 - precision.
Accuracy is the fraction of samples that the classifier correctly predicted (both positive and negative) to the total number of samples in test data set. Accuracy makes no  distinction between classes; correct answers for both positive and negative cases are treated equally. When cost of misclassification is different or if there are a lot more test data of one class than the other, then accuracy would give a very distorted picture, because the class with more examples will dominate the statistic. Bike buyers data set is balanced, it contains 9132 positive and 9352 negative examples.
In contrast to accuracy, Matthews correlation coefficient MCC is generally regarded as a balanced measure of the quality of binary classifications if the classes are of very different sizes.
Mutual combination of precision and recall is called F-Measure or F-Score 2PR/(P+R) and is another metric useful for rating classification accuracy and comparing different classification models or algorithms. F-Score equal to 1 represents model with perfect precision and sensitivity whilst F-Score equal to 0 is the opposite.
  val stats = Stats(confusionMatrix(predictionsAndLabels))
  println(stats.toString)

Example output can look like that:

TP: 816.0, TN: 807.0, FP: 123.0, FN: 114.0 
TPR (recall/sensitivity): 0.8774193548387097 
TNR (specificity): 0.867741935483871 
PPV (precision): 0.8690095846645367 
NPV: 0.8762214983713354 
FPR (fall-out): 0.13225806451612898 
FNR: 0.1225806451612903 
FDR: 0.1309904153354633 
ACC (accuracy): 0.8725806451612903 
F1 (F-Measure): 0.8731942215088282 
MCC (Matthews correlation coefficient): 0.745196185862156
Some of described metrics and some additional are available in Spark’s binary metrics evaluator:
    val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
The ROC curve (receiver operating characteristic) shows the sensitivity of the classifier by plotting the rate of true positives to the rate of false positives. In other words, the perfect classifier that makes no mistakes would hit a true positive rate of 1, without incurring any false positives. Calculating area under ROC curve allows to express this relation as single number, which with caution can be used for model comparison. High value of AUC can be treated as representation of good classification model, which reflects a lot of space under curve, when it goes to point of perfect classification. Low value of AUC is the opposite. In Spark ROC curve is available in form of RDD containing (false positive rate, true positive rate) with (0.0, 0.0) prepended and (1.0, 1.0) appended to it, and area under ROC curve is available as single value.
    val roc = metrics.roc
    val auROC = metrics.areaUnderROC
Precision-Recall curve (Spark returns it in reverse order) is available along with area under PR curve
    val PR = metrics.pr
    val auPR = metrics.areaUnderPR
Values of precision, recall and f-measure can be also read from BinaryClassificationMetrics object. In this case, Spark calculates them for both classes separately:
    val precision = metrics.precisionByThreshold
    precision.foreach {
      case (t, p) =>
        println(s"Threshold: $t, Precision: $p")
    }

    val recall = metrics.recallByThreshold
    recall.foreach {
      case (t, r) =>
        println(s"Threshold: $t, Recall: $r")
    }
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach {
      case (t, f) =>
        println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

Example output can look like that:
Threshold: 1.0, Precision: 0.8690095846645367
Threshold: 0.0, Precision: 0.5
Threshold: 1.0, Recall: 0.8774193548387097
Threshold: 0.0, Recall: 1.0
Threshold: 1.0, F-score: 0.8731942215088282, Beta = 1
Threshold: 0.0, F-score: 0.6666666666666666, Beta = 1
Threshold: 1.0, F-score: 0.8731942215088282, Beta = 0.5
Threshold: 0.0, F-score: 0.6666666666666666, Beta = 0.5
Area under PR (precision-recall curve) = 0.9038596310419458
Area under ROC (Receiver Operating Characteristic) = 0.8725806451612903

