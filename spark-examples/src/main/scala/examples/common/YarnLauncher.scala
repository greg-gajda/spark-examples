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

import org.apache.spark.launcher.SparkLauncher
import java.util.concurrent.Executors

import collection.JavaConversions._
object YarnLauncher {
  
  val mode = "yarn-client"
  val mainClass = "examples.regression.HousePricesPrediction"
  
  def main(args: Array[String]): Unit = {
    val launcher = new SparkLauncher()
      .setAppResource("build/libs/spark-examples-1.0.jar")
      .setMainClass(mainClass)
      .setMaster(mode)
      .launch();
    
    
    val tf = Executors.defaultThreadFactory()
    tf.newThread(new RunnableInputStreamReader(launcher.getInputStream(), "input")).start()
    tf.newThread(new RunnableInputStreamReader(launcher.getErrorStream(), "error")).start()
    
    println("Executing ...")
    val exitCode = launcher.waitFor()
    println(s"Finished, exit code: $exitCode")
    
  }
}