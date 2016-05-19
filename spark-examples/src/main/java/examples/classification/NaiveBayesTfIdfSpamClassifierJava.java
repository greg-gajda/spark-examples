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

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;
import scala.Tuple3;

public class NaiveBayesTfIdfSpamClassifierJava {
	
	static void evaluateModel(NaiveBayesModel model, JavaRDD<LabeledPoint> test){
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
	
	public static void main(String [] args){
		try (JavaSparkContext sc = new JavaSparkContext(configLocalMode("NaiveBayes exploiting TFIDF for spam classification in Java 8"))) {
			HashingTF hash = new HashingTF(100000);
			JavaRDD<String> file = localFile("sms-labeled.txt", sc);			
			JavaRDD<Tuple3<String, List<String>, Vector>> raw = file.distinct().map(
				s -> s.split("\\t+")
			).map(
				a -> new Tuple2<>(a[0], Arrays.stream(a[1].split("\\s+")).map(w -> w.toLowerCase()).collect(Collectors.toList()))
			).map(
				t -> new Tuple3<>(t._1, t._2, hash.transform(t._2))
			).cache();
			
			IDFModel idf = new IDF().fit(raw.map(t -> t._3()).rdd());
			JavaRDD<LabeledPoint> data = raw.map(t -> {
				int label = 0;
				if(t._1().equals("spam")){
					label = 1;
				}
				return new LabeledPoint(label, idf.transform(t._3()));
			});
			
			JavaRDD<LabeledPoint>[] split = data.randomSplit(new double[] { .8, .2 });
			JavaRDD<LabeledPoint> train = split[0].cache();
			JavaRDD<LabeledPoint> test = split[1].cache();
			
			NaiveBayesModel model = NaiveBayes.train(train.rdd());
			evaluateModel(model, test);			
		}
	}
}
