package examples.regression;

import java.util.Collections;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;

public class HousePricesPredictionJava {

	static LinearRegressionModel createLinearRegressionModel(JavaRDD<LabeledPoint> rdd, Integer numIterations,
			Double stepSize) {
		return LinearRegressionWithSGD.train(rdd.rdd(),
				numIterations == null ? 100 : numIterations,
				stepSize == null ? 0.01 : stepSize);
	}

	static DecisionTreeModel createDecisionTreeRegressionModel(JavaRDD<LabeledPoint> rdd, Integer maxDepth,
			Integer maxBins) {
		String impurity = "variance";
		return DecisionTree.trainRegressor(rdd,
				Collections.emptyMap(),
				impurity, maxDepth == null ? 10 : maxDepth,
				maxBins == null ? 20 : maxBins);
	}

	public static void main(String[] args) {

		org.apache.log4j.PropertyConfigurator.configure(Thread.currentThread().getContextClassLoader().getResourceAsStream("log4j.config"));

		SparkSession spark = SparkSession
			.builder()
			.master("local[*]")
			.appName("Regression for House prices predictions in Java 8")
			.getOrCreate();

		try (JavaSparkContext sc = new JavaSparkContext(spark.sparkContext())) {
			JavaRDD<String> hdFile = sc.textFile(args.length == 0 ? "data/house-data.csv" : args[0] + "house-data.csv");

			JavaRDD<LabeledPoint> houses = hdFile.map(s -> s.split(",")).
				filter(t -> !"date".equals(t[1])).
				map(a -> new HouseModelJava(a).toLabeledPoint());

			StandardScalerModel scaler = new StandardScaler(true, true).fit(houses.map(dp -> dp.features()).rdd());

			JavaRDD<LabeledPoint>[] split = houses.map(
					dp -> new LabeledPoint(dp.label(), scaler.transform(dp.features())))
				.randomSplit(new double[] { .9, .1 }, 10204L);
			JavaRDD<LabeledPoint> train = split[0].cache();
			JavaRDD<LabeledPoint> test = split[1].cache();

			DecisionTreeModel model = createDecisionTreeRegressionModel(train, null, null);

			test.take(5).stream().forEach(
					x -> System.out.println(String.format("Predicted: %.1f, Label: %.1f",
							model.predict(x.features()), x.label()))
				);

			JavaPairRDD<Object, Object> predictionsAndValues = test.mapToPair(
				p -> new Tuple2<Object, Object>(model.predict(p.features()), p.label())
				);

			System.out.println("Mean house price: " + test.mapToDouble(x -> x.label()).mean());
			System.out.println("Max prediction error: " + predictionsAndValues.mapToDouble(
					t2 -> Math.abs(Double.class.cast(t2._2) - Double.class.cast(t2._1))).max());

			RegressionMetrics metrics = new RegressionMetrics(predictionsAndValues.rdd());

			System.out.println(String.format("Mean Squared Error: %.2f", metrics.meanSquaredError()));
			System.out.println(String.format("Root Mean Squared Error: %.2f", metrics.rootMeanSquaredError()));
			System.out.println(String.format("Coefficient of Determination R-squared: %.2f", metrics.r2()));
			System.out.println(String.format("Mean Absoloute Error: %.2f", metrics.meanAbsoluteError()));
			System.out.println(String.format("Explained variance: %.2f", metrics.explainedVariance()));
		}
	}
}
