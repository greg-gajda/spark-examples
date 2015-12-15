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

import org.apache.spark.rdd.RDD

class Stats(val tp: Double, val tn: Double, val fp: Double, val fn: Double) {

  /**
   * recall/sensitivity or true positive rate (TPR)
   */
  val TPR = tp / (tp + fn)
  val recall = TPR
  val sensitivity = TPR

  /**
   * specificity or true negative rate
   */
  val TNR = tn / (tn + fp)
  val specificity = TNR

  /**
   * precision or positive predictive value
   */
  val PPV = tp / (tp + fp)
  val precision = PPV

  /**
   * negative predictive value
   */
  val NPV = tn / (tn + fn)

  /**
   * fall-out or false positive rate
   */
  val FPR = 1.0 - specificity

  /**
   * false negative rate
   */
  val FNR = 1.0 - recall

  /**
   * false discovery rate
   */
  val FDR = 1.0 - precision

  /**
   * accuracy
   */
  val ACC = (tp + tn) / (tp + fp + fn + tn)
  val accuracy = ACC

  val F1 = 2 * PPV * TPR / (PPV + TPR)

  /**
   * Matthews correlation coefficient
   */
  val MCC = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (fp + tn) * (tn + fn))

  override def toString = {
    s"TP: $tp, TN: $tn, FP: $fp, FN: $fn \n" +
      s"TPR (recall/sensitivity): $TPR \n" +
      s"TNR (specificity): $TNR \n" +
      s"PPV (precision): $PPV \n" +
      s"NPV: $NPV \n" +
      s"FPR (fall-out): $FPR \n" +
      s"FNR: $FNR \n" +
      s"FDR: $FDR \n" +
      s"ACC (accuracy): $ACC \n" +
      s"F1 (F-Measure): $F1 \n" +
      s"MCC (Matthews correlation coefficient): $MCC "
  }
}

object Stats {

  def apply(cc: (Int, Int, Int, Int)): Stats = new Stats(cc._1, cc._2, cc._3, cc._4)

  def confusionMatrix(rdd: RDD[(Double, Double)]) = {
    rdd.aggregate((0, 0, 0, 0))(
      seqOp = (t, pal) => {
        val (tp, tn, fp, fn) = t
        (if (pal._1 == pal._2 && pal._2 == 1.0) tp + 1 else tp,
          if (pal._1 == pal._2 && pal._2 == 0.0) tn + 1 else tn,
          if (pal._1 == 1.0 && pal._2 == 0.0) fp + 1 else fp,
          if (pal._1 == 0.0 && pal._2 == 1.0) fn + 1 else fn)
      },
      combOp = (t1, t2) => t1 + t2)
  }

  implicit class Tupple4Add[A: Numeric, B: Numeric, C: Numeric, D: Numeric](t: (A, B, C, D)) {
    import Numeric.Implicits._
    def +(p: (A, B, C, D)) = (p._1 + t._1, p._2 + t._2, p._3 + t._3, p._4 + t._4)
  }

} 