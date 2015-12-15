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
package examples.cassandra

import java.net.InetAddress
import java.io.InputStream
import scala.io.Source
import com.datastax.driver.core.querybuilder.QueryBuilder
import scala.collection.JavaConversions._
import com.datastax.driver.core.DataType

object LoadBikeBuyers {
  def main(args: Array[String]): Unit = {

    org.apache.log4j.BasicConfigurator.configure()
    val host = args(0) 
    val cc = com.datastax.spark.connector.cql.CassandraConnector(Set(InetAddress.getByName(host)))

    val keyspaceCql = Source.fromInputStream(getClass.getResourceAsStream("/create_spark_keyspace.cql")).mkString
    val tableCql = Source.fromInputStream(getClass.getResourceAsStream("/create_bike_buyers_table.cql")).mkString

    val bbFile = Source.fromFile("data/bike-buyers.txt", "utf8").getLines()

    cc.withSessionDo(s => {
      s.execute(keyspaceCql)
      s.execute(tableCql)

      val columns = s.getCluster.getMetadata.getKeyspace("spark").getTable("bike_buyers").getColumns

      bbFile.map { x => x.split("\\t") }.foreach(row => {

        val insert = new QueryBuilder(s.getCluster).insertInto("spark", "bike_buyers")
        row.zipWithIndex.foreach(vi => {
          val column = columns(vi._2)
          insert.value(column.getName,
            if (column.getType == DataType.text()) vi._1
            else if (column.getType == DataType.cfloat()) vi._1.replaceFirst(",", ".").toFloat
            else vi._1.toInt)
        })
        s.execute(insert)
      })
    })
  }
}