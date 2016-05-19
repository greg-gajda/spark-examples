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

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import examples.PrintUtils.printMetrics
import examples.classification.Stats.confusionMatrix
import examples.common.Application._
import org.apache.spark.mllib.feature.IDF

object NaiveBayesTfIdfSpamClassifier {

  def evaluateModel(model: NaiveBayesModel, test: RDD[LabeledPoint]) = {
    val predict = model.predict(test.map(_.features))

    test.take(5).foreach {
      x => println(s"Predicted: ${model.predict(x.features)}, Label: ${x.label}")
    }

    val predictionsAndLabels = test.map {
      point => (model.predict(point.features), point.label)
    }

    val stats = Stats(confusionMatrix(predictionsAndLabels))
    println(stats.toString)

    val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
    printMetrics(metrics)
  }

  def tfidf(data: RDD[(String, Array[String])])(implicit sc: SparkContext) = {
    val docs = data.count.toDouble
    //TF - terms frequencies 
    val tfs = data.map {
      t => (t._1, t._2.foldLeft(Map.empty[String, Int])((m, s) => m + (s -> (1 + m.getOrElse(s, 0)))))
    }

    //TF-IDF
    val idfs = data.flatMap(_._2).map((_, 1)).reduceByKey(_ + _).map {
      case (term, count) => (term, math.log(docs / (1 + count)))
    }.collectAsMap

    //idfs.lookup("").lift(0).getOrElse(0d)
    tfs.map {
      case (m, tf) =>
        (m, tf.map {
          case (term, freq) => (term, freq * idfs.getOrElse(term, 0d))
        })
    }
  }

  def main(args: Array[String]): Unit = {

    implicit val sc = new SparkContext(configLocalMode("NaiveBayes exploiting TFIDF for spam classification"))

    val hash = new HashingTF(numFeatures = 100000)
    val raw = sc.textFile("data/sms-labeled.txt").distinct().map {
      _.split("\\t+")
    }.map {
      a => (a(0), a(1).split("\\s+").map(_.toLowerCase()))
    }.map {
      t => (t._1, t._2, hash.transform(t._2))
    }.cache

    val idf = new IDF().fit(raw.map(_._3))
    val data = raw.map {
      t => LabeledPoint(if (t._1 == "spam") 1 else 0, idf.transform(t._3))
    }

    val Array(train, test) = data.randomSplit(Array(.8, .2), 102059L)
    val model = NaiveBayes.train(train)
    evaluateModel(model, test)

    val termsInSpamMsgs = tfidf(raw.filter(_._1 == "spam").map(t => (t._1, t._2))).sortBy(_._2.values, ascending = false)
    termsInSpamMsgs.take(10).foreach(println)

    sc.stop()
  }
}