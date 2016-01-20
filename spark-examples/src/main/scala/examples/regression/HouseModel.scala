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
package examples.regression

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

import examples.common.LabeledPointConverter

case class HouseModel(id: Long,
                      date: java.sql.Date,
                      price: Double,
                      bedrooms: Int,
                      bathrooms: Double,
                      sqft_living: Int,
                      sqft_lot: Int,
                      floors: Double,
                      waterfront: Int,
                      view: Int,
                      condition: Int,
                      grade: Int,
                      sqft_above: Int,
                      sqft_basement: Int,
                      yr_built: Int,
                      yr_renovated: Int,
                      zipcode: String,
                      lat: Double,
                      long: Double,
                      sqft_living15: Int,
                      sqft_lot15: Int)
    extends LabeledPointConverter {

  def label() = price
  def features() = HouseModel.convert(this)
}

object HouseModel {

  def df = new java.text.SimpleDateFormat("yyyyMMdd'T'hhmmss")

  def apply(row: Array[String]) = new HouseModel(
    row(0).toLong, new java.sql.Date(df.parse(row(1)).getTime),
    row(2).toDouble, row(3).toInt,
    row(4).toDouble, row(5).toInt, row(6).toInt,
    row(7).toDouble, row(8).toInt, row(9).toInt,
    row(10).toInt, row(11).toInt, row(12).toInt,
    row(13).toInt, row(14).toInt, row(15).toInt, row(16),
    row(17).toDouble, row(18).toDouble, row(19).toInt, row(20).toInt)

  def convert(model: HouseModel) = Vectors.dense(
    model.id.toDouble,
    model.bedrooms.toDouble,
    model.bathrooms,
    model.sqft_living.toDouble,
    model.sqft_lot.toDouble,
    model.floors,
    model.waterfront.toDouble,
    model.view.toDouble,
    model.condition.toDouble,
    model.grade.toDouble,
    model.sqft_above.toDouble,
    model.sqft_basement.toDouble,
    model.yr_built.toDouble,
    model.yr_renovated.toDouble,
    model.lat,
    model.long,
    model.sqft_living15.toDouble,
    model.sqft_lot15.toDouble)
}