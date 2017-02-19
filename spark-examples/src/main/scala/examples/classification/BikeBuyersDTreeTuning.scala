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
package examples.classification

import scala.Range
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.DecisionTree
import examples.common.Application.configLocalMode
import examples.common.DataLoader.localFile
import examples.classification.Stats.confusionMatrix
import org.apache.spark.sql.SparkSession

object BikeBuyersDTreeTuning {
  
  def main(args: Array[String]): Unit = {  
    org.apache.log4j.PropertyConfigurator.configure(Thread.currentThread().getContextClassLoader().getResourceAsStream("log4j.config"))
    
    val spark = SparkSession.builder().appName("Classification of Bike Buyers with Decision Tree Tunning").master("local[*]").getOrCreate()
    val sc = spark.sparkContext
    
    val bbFile = sc.textFile(args.headOption.getOrElse("data/") + "bike-buyers.txt")
    
    val data = bbFile.map { row =>
      BikeBuyerModel(row.split("\\t")).toLabeledPoint
    }

    val Array(train, test) = data.randomSplit(Array(.9, .1), 102059L)
    train.cache()
    test.cache()

    val numClasses = 2
    val tuning =
      for (
        impurity <- Array("entropy", "gini");
        maxDepth <- Range(5, 25, 5);
        maxBins <- Range(10, 50, 2)
      ) yield {
        val model = DecisionTree.trainClassifier(
          train,
          numClasses,
          BikeBuyerModel.categoricalFeaturesInfo(),
          impurity,
          maxDepth,
          maxBins)
        val predictionsAndLabels = test.map {
          point => (model.predict(point.features), point.label)
        }
        val stats = Stats(confusionMatrix(predictionsAndLabels))
        val metrics = new BinaryClassificationMetrics(predictionsAndLabels)        
        ((impurity, maxDepth, maxBins), stats.MCC, stats.ACC, metrics.areaUnderPR, metrics.areaUnderROC)
      }
    tuning.sortBy(_._2).reverse.foreach{
      x => println(x._1 + ", Matthews correlation coefficient: " + x._2 + ", Accuracy: " + x._3+ ", Area under PR: " + x._4+ ", Area under ROC: " + x._5)
    }

    spark.close
  }
}