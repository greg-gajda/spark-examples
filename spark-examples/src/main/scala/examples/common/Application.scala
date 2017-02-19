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

import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import org.apache.log4j.Level

@deprecated("this object is to be removed", "spark-examples")
object Application {
  
  def configLocalMode(appName: String): SparkConf = {
    val conf = new SparkConf().setAppName(appName)
    conf.setMaster("local[*]")
    conf.set("spark.cassandra.connection.host", cassandraHost)
    conf
  }

  def configStandaloneClusterMode(appName: String): SparkConf = {
    val conf = new SparkConf().setAppName(appName)
    conf.setMaster(sparkMaster)
    conf.setJars(Array("build/libs/spark-examples-1.0.jar"))
    conf.set("spark.cassandra.connection.host", cassandraHost)
    conf
  }

  def configYarnClientMode(appName: String): SparkConf = {
    val conf = new SparkConf().setAppName(appName)
    conf.setMaster("yarn-client")
    conf.setJars(Array("build/libs/spark-examples-1.0.jar"))
    conf.set("spark.cassandra.connection.host", cassandraHost)
    conf
  }
  
  def configYarnClusterMode(appName: String): SparkConf = {
    val conf = new SparkConf().setAppName(appName)
    conf.setMaster("yarn-cluster")
    conf.setJars(Array("build/libs/spark-examples-1.0.jar"))
    conf.set("spark.cassandra.connection.host", cassandraHost)
    conf
  }
    
  val sparkMaster = "spark://192.168.1.15:7077"
  val cassandraHost = "192.168.1.34"
}