Java 8 version on binary classification by Random Forest:

```java
try (JavaSparkContext sc = new JavaSparkContext(configLocalMode())) {
	JavaRDD<String> bbFile = localFile(sc);

	JavaRDD<LabeledPoint> data = bbFile.map(r -> new BikeBuyerModelJava(r.split("\\t")).toLabeledPoint());
	JavaRDD<LabeledPoint>[] split = data.randomSplit(new double[] { .9, .1 });
	JavaRDD<LabeledPoint> train = split[0].cache();
	JavaRDD<LabeledPoint> test = split[1].cache();

	Integer numClasses = 2;
    Integer numTrees = 10;
    String featureSubsetStrategy = "auto";
	String impurity = "entropy";
	Integer maxDepth = 20;
	Integer maxBins = 34;
	Integer seed = 12345;

	final RandomForestModel model = RandomForest.trainClassifier(train, numClasses, BikeBuyerModelJava.categoricalFeaturesInfo(), numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

    test.take(5).forEach(x -> {
    	System.out.println(String.format("Predicted: %.1f, Label: %.1f", model.predict(x.features()), x.label()));	
    });

    JavaPairRDD<Object, Object> predictionsAndLabels = test.mapToPair(
	    p -> new Tuple2<Object, Object>(model.predict(p.features()), p.label())
    );
	
    Stats stats = Stats.apply(confusionMatrix(predictionsAndLabels.rdd()));
    System.out.println(stats.toString());
	
	BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionsAndLabels.rdd());
	printMetrics(metrics);
}
```