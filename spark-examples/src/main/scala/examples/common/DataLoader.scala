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
package examples.common

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

@deprecated("this object is to be removed", "spark-examples")
object DataLoader {
  type LOAD = SparkContext => RDD[String]

  def localFile(fileName: String): LOAD = sc => {
    sc.textFile("data/" + fileName)
  }

  def hdfsFile(fileName: String): (SparkContext => RDD[String]) = sc => {
    sc.textFile("hdfs://192.168.1.34:19000/spark/" + fileName)
  }

  def cassandraFile: LOAD = sc => {
    import com.datastax.spark.connector._
    sc.cassandraTable("spark", "bike_buyers").map { row =>
      row.columnValues.mkString("\t")
    }
  }

}