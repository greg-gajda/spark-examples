There are some differences in using Spark from Java. 
The most important is that instead of SparkContext its java friendly version called JavaSparkContext must be used. 
Methods of this class returns java wrappers of RDD objects (JavaRDD) and works with Java collections. 
To create JavaSparkContext it is convenient to use try-with-resources statement:  
```
  try (JavaSparkContext sc = new JavaSparkContext(configLocalMode())) {…}
```
JavaSparkContext class implements Closeable interface and calls stop method:
```
  override def close(): Unit = stop()
```
Functions to load data from local storage, HDFS and Cassandra go as follow.

Again, to run on cluster, local file should be located on network storage available for every Spark’s workers or copied to exactly the same location on every node.
```
public static JavaRDD<String> localFile(JavaSparkContext sc) {
    return sc.textFile("data/bike-buyers");
}
```
For Hadoop, HDFS URL must be provided:
```
public static JavaRDD<String> hdfsFile(JavaSparkContext sc) {
    return sc.textFile("hdfs://192.168.1.15:9000/spark/bike-buyers");
}
```
Access to Cassandra is a bit more complicated. 
First map function converts key value pair into Spark’s Tuple2 containing position of column and value converted to String. 
Then Stream is sorted by column position. Second map function takes String value from Tuple2, which finally is reduced to single line. 
```
import static com.datastax.spark.connector.japi.CassandraJavaUtil.javaFunctions;

public static JavaRDD<String> cassandraFile(JavaSparkContext sc) {
    return javaFunctions(sc).cassandraTable("spark", "bike_buyers").map(
		row -> {
			return row.toMap().entrySet().stream().
			map(e -> 
                new Tuple2<>(row.indexOf(e.getKey()), e.getValue().toString())
            ).sorted((t1, t2) -> t1._1().compareTo(t2._1())).
            map(t -> t._2()).reduce((a, b) -> a + "\t" + b).get();
	});
  }
```
Data conversion into LabeledPoint data structure can be done with little help from class that reflects raw data structure, and makes conversion into LabeledPoints. 
Here Java is mixed with Scala, BikeBuyerModelJava implements Scala trait. 
Browsing Spark’s source code shows many places with similar approach. 
```
public class BikeBuyerModelJava implements LabeledPointConverter {

	private final Integer customerKey;
	private final Integer age;
	private final Integer bikeBuyer;
	private final String commuteDistance;
	private final String englishEducation;
	private final String gender;
	private final Integer houseOwnerFlag;
	private final String maritalStatus;
	private final Integer numberCarsOwned;
	private final Integer numberChildrenAtHome;
	private final String englishOccupation;
	private final String region;
	private final Integer totalChildren;
	private final Float yearlyIncome;

	public BikeBuyerModelJava(Integer customerKey, Integer age, Integer bikeBuyer, String commuteDistance,
			String englishEducation, String gender, Integer houseOwnerFlag, String maritalStatus,
			Integer numberCarsOwned, Integer numberChildrenAtHome, String englishOccupation, String region,
			Integer totalChildren, Float yearlyIncome) {
		super();
		this.customerKey = customerKey;
		this.age = age;
		this.bikeBuyer = bikeBuyer;
		this.commuteDistance = commuteDistance;
		this.englishEducation = englishEducation;
		this.gender = gender;
		this.houseOwnerFlag = houseOwnerFlag;
		this.maritalStatus = maritalStatus;
		this.numberCarsOwned = numberCarsOwned;
		this.numberChildrenAtHome = numberChildrenAtHome;
		this.englishOccupation = englishOccupation;
		this.region = region;
		this.totalChildren = totalChildren;
		this.yearlyIncome = yearlyIncome;
	}

	public BikeBuyerModelJava(String... row) {
		this(Integer.valueOf(row[0]), Integer.valueOf(row[1]), Integer.valueOf(row[2]), row[3], row[4], row[5], Integer
				.valueOf(row[6]), row[7], Integer.valueOf(row[8]), Integer.valueOf(row[9]), row[10], row[11], Integer
				.valueOf(row[12]), Float.valueOf(row[13].replaceFirst(",", ".")));
	}

	@Override
	public LabeledPoint toLabeledPoint() {
		return new LabeledPoint(label(), features());
	}
	
	@Override
	public double label() {
		return bikeBuyer.doubleValue();
	}

	@Override
	public Vector features() {
		double[] features = new double[getClass().getDeclaredFields().length - 1];
		features[0] = customerKey.doubleValue();
		features[1] = age.doubleValue();
		switch (commuteDistance) {
		case "0-1 Miles":
			features[2] = 0d;
			break;
		case "1-2 Miles":
			features[2] = 1d;
			break;
		case "2-5 Miles":
			features[2] = 2d;
			break;
		case "5-10 Miles":
			features[2] = 3d;
			break;
		case "10+ Miles":
			features[2] = 4d;
			break;
		default:
		}
		switch (englishEducation) {
		case "High School":
			features[3] = 0d;
			break;
		case "Partial High School":
			features[3] = 1d;
			break;
		case "Partial College":
			features[3] = 2d;
			break;
		case "Graduate Degree":
			features[3] = 3d;
			break;
		case "Bachelors":
			features[3] = 4d;
			break;
		default:
		}
		switch (gender) {
		case "M":
			features[4] = 0d;
			break;
		case "F":
			features[4] = 1d;
			break;
		default:
		}
		features[5] = houseOwnerFlag.doubleValue();
		switch (maritalStatus) {
		case "S":
			features[6] = 0d;
			break;
		case "M":
			features[6] = 1d;
			break;
		default:
		}
		features[7] = numberCarsOwned.doubleValue();
		features[8] = numberChildrenAtHome.doubleValue();
		switch (englishOccupation) {
		case "Professional":
			features[9] = 0d;
			break;
		case "Clerical":
			features[9] = 1d;
			break;
		case "Manual":
			features[9] = 2d;
			break;
		case "Management":
			features[9] = 3d;
			break;
		case "Skilled Manual":
			features[9] = 4d;
			break;
		default:
		}
		switch (region) {
		case "North America":
			features[10] = 0d;
			break;
		case "Pacific":
			features[10] = 1d;
			break;
		case "Europe":
			features[10] = 2d;
			break;
		default:
		}
		features[11] = totalChildren.doubleValue();
		features[12] = yearlyIncome;
		return Vectors.dense(features);
	}

	public static Map<Integer, Integer> categoricalFeaturesInfo() {
		return new HashMap<Integer, Integer>() {
			private static final long serialVersionUID = 1L;
			{
				put(2, 5);
				put(3, 5);
				put(4, 2);
				put(6, 2);
				put(9, 5);
				put(10, 3);
			}
		};
	}
	//and getters
}
```
Now, data in format required by Spark can be acquired:
```
JavaRDD<LabeledPoint> data = bbFile.map(r -> 
new BikeBuyerModelJava(r.split("\\t")).toLabeledPoint()
);
```
Splitting data set for training and testing also looks a bit different in Java:
```
  JavaRDD<LabeledPoint>[] split = data.randomSplit( new double[] { .9, .1 } );
  JavaRDD<LabeledPoint> train = split[0].cache();
  JavaRDD<LabeledPoint> test = split[1].cache();
```
Now classification model can be built:
```
  Integer numClasses = 2;
  String impurity = "entropy";
  Integer maxDepth = 20;
  Integer maxBins = 34;

  final DecisionTreeModel dtree = DecisionTree.trainClassifier(train, numClasses, BikeBuyerModelJava.categoricalFeaturesInfo(), impurity, maxDepth, maxBins); 
```
After displaying first 5 predictions:
```
test.take(5).forEach(x -> { 
  System.out.println(String.format("Predicted: %.1f, Label: %.1f", dtree.predict(x.features()), x.label()));	
});
```
With sample output:

Predicted: 1,0, Label: 1,0
Predicted: 1,0, Label: 1,0
Predicted: 1,0, Label: 1,0
Predicted: 1,0, Label: 1,0
Predicted: 0,0, Label: 0,0

Metrics can be evaluated:
```
JavaRDD<Tuple2<Object, Object>> predictionsAndLabels = test.map(p -> 
	new Tuple2<Object, Object>(dtree.predict(p.features()), p.label())
);
  
Stats stats = Stats.apply(confusionMatrix(predictionsAndLabels.rdd()));
System.out.println(stats.toString());

BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionsAndLabels.rdd());
printMetrics(metrics);
```