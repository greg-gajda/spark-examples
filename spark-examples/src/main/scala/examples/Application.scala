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
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Application {
  org.apache.log4j.BasicConfigurator.configure()
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  
  def configLocalMode: SparkConf = {
    val conf = new SparkConf().setAppName(applicationName)
    conf.setMaster("local[*]")
    conf.set("spark.cassandra.connection.host", cassandraHost)
    conf
  }

  def configStandaloneClusterMode: SparkConf = {
    val conf = new SparkConf().setAppName(applicationName)
    conf.setMaster("spark://192.168.1.15:7077")
    conf.setJars(Array("build/libs/spark-examples-1.0.jar"))
    conf.set("spark.cassandra.connection.host", cassandraHost)
    conf
  }

  val applicationName = "Decision Tree Algorithm as classifier of Bike Buyers"
  val cassandraHost = "127.0.0.1"
}