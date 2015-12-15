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

package examples

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object PrintUtils {

  def printMetrics(metrics: BinaryClassificationMetrics) = {
    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach {
      case (t, p) =>
        println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach {
      case (t, r) =>
        println(s"Threshold: $t, Recall: $r")
    }

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach {
      case (t, f) =>
        println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach {
      case (t, f) =>
        println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPR
    val auPR = metrics.areaUnderPR
    println("Area under PR (precision-recall curve) = " + auPR)

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC (Receiver Operating Characteristic) = " + auROC)
  }

}