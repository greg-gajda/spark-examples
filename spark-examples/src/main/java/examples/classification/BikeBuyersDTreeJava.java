/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package examples.classification;

import static examples.common.Application.configLocalMode;
import static examples.common.FilesLoaderJava.localFile;
import static examples.PrintUtils.printMetrics;
import static examples.classification.Stats.confusionMatrix;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

import scala.Tuple2;

public class BikeBuyersDTreeJava {

	public static void main(String[] args) {
		
		try (JavaSparkContext sc = new JavaSparkContext(configLocalMode("Classification of Bike Buyers with DecisionTree in Java 8"))) {
			JavaRDD<String> bbFile = localFile("bike-buyers.txt", sc);

			JavaRDD<LabeledPoint> data = bbFile.map(r -> new BikeBuyerModelJava(r.split("\\t")).toLabeledPoint());
			JavaRDD<LabeledPoint>[] split = data.randomSplit(new double[] { .9, .1 });
			JavaRDD<LabeledPoint> train = split[0].cache();
			JavaRDD<LabeledPoint> test = split[1].cache();

			Integer numClasses = 2;
			String impurity = "entropy";
			Integer maxDepth = 20;
			Integer maxBins = 34;

			final DecisionTreeModel dtree = DecisionTree.trainClassifier(train, numClasses,
					BikeBuyerModelJava.categoricalFeaturesInfo(), impurity, maxDepth, maxBins);

		    test.take(5).forEach(x -> {
		    	System.out.println(String.format("Predicted: %.1f, Label: %.1f", dtree.predict(x.features()), x.label()));	
		    });

		    JavaPairRDD<Object, Object> predictionsAndLabels = test.mapToPair(
		    	p -> new Tuple2<Object, Object>(dtree.predict(p.features()), p.label())
		    );
			
		    Stats stats = Stats.apply(confusionMatrix(predictionsAndLabels.rdd()));
		    System.out.println(stats.toString());
			
			BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionsAndLabels.rdd());
			printMetrics(metrics);
		}
	}

}
