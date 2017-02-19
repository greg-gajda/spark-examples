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
package examples.common;

import static com.datastax.spark.connector.japi.CassandraJavaUtil.javaFunctions;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

@Deprecated
public class FilesLoaderJava {

	public static JavaRDD<String> localFile(String fileName, JavaSparkContext sc) {
		return sc.textFile("data/".concat(fileName));
	}

	public static JavaRDD<String> hdfsFile(String fileName, JavaSparkContext sc) {
		return sc.textFile("hdfs://192.168.1.15:9000/spark/".concat(fileName));
	}

	public static JavaRDD<String> cassandraFile(JavaSparkContext sc) {
		return javaFunctions(sc).cassandraTable("spark", "bike_buyers").map(
				row -> {
					return row.toMap().entrySet().stream()
							.map(e -> new Tuple2<>(row.indexOf(e.getKey()), e.getValue().toString()))
							.sorted((t1, t2) -> t1._1().compareTo(t2._1())).map(t -> t._2())
							.reduce((a, b) -> a + "\t" + b).get();
				});

	}

}
