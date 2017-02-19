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
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.InputStream

object YarnLauncher {
  
  def main(args: Array[String]): Unit = {
    val launcher = new SparkLauncher()
      .setAppResource("build/libs/spark-examples-1.0.jar")
      .setMainClass(args(0))
      .setMaster(args(1))
      .setDeployMode(args(2))
      .launch();
    
    
    val tf = Executors.defaultThreadFactory()
    tf.newThread(new RunnableInputStreamReader(launcher.getInputStream(), "input")).start()
    tf.newThread(new RunnableInputStreamReader(launcher.getErrorStream(), "error")).start()
    
    println("Executing ...")
    val exitCode = launcher.waitFor()
    println(s"Finished, exit code: $exitCode")
    
  }
}

class RunnableInputStreamReader(is: InputStream, name: String) extends Runnable {

  val reader = new BufferedReader(new InputStreamReader(is))

  def run() = {
    var line = reader.readLine();
    while (line != null) {
      System.out.println(line);
      line = reader.readLine();
    }
    reader.close();
  }
  
}