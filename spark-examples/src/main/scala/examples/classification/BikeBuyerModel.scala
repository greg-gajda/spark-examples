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

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import examples.common.LabeledPointConverter

case class BikeBuyerModel(customerKey: Int,
                          age: Int, 
                          bikeBuyer: Int, 
                          commuteDistance: String,
                          englishEducation: String, 
                          gender: String, 
                          houseOwnerFlag: Int,
                          maritalStatus: String, 
                          numberCarsOwned: Int, 
                          numberChildrenAtHome: Int,
                          englishOccupation: String, 
                          region: String, 
                          totalChildren: Int, 
                          yearlyIncome: Float)
    extends LabeledPointConverter {

  def label() = bikeBuyer.toDouble
  def features() = BikeBuyerModel.convert(this)
}

object BikeBuyerModel {

  def apply(row: Array[String]) = new BikeBuyerModel(
    row(0).toInt, row(1).toInt, row(2).toInt, row(3),
    row(4), row(5), row(6).toInt,
    row(7), row(8).toInt, row(9).toInt,
    row(10), row(11), row(12).toInt, row(13).replaceFirst(",", ".").toFloat)

  def categoricalFeaturesInfo() = {
    Map[Int, Int](2 -> 5, 3 -> 5, 4 -> 2, 6 -> 2, 9 -> 5, 10 -> 3)
  }

  def convert(model: BikeBuyerModel) = Vectors.dense(
    model.customerKey.toDouble,
    model.age.toDouble,
    model.commuteDistance match {
      case "0-1 Miles"  => 0d
      case "1-2 Miles"  => 1d
      case "2-5 Miles"  => 2d
      case "5-10 Miles" => 3d
      case "10+ Miles"  => 4d
    },
    model.englishEducation match {
      case "High School"         => 0d
      case "Partial High School" => 1d
      case "Partial College"     => 2d
      case "Graduate Degree"     => 3d
      case "Bachelors"           => 4d
    },
    model.gender match {
      case "M" => 0d
      case "F" => 1d
    },
    model.houseOwnerFlag.toDouble,
    model.maritalStatus match {
      case "S" => 0d
      case "M" => 1d
    },
    model.numberCarsOwned.toDouble,
    model.numberChildrenAtHome.toDouble,
    model.englishOccupation match {
      case "Professional"   => 0d
      case "Clerical"       => 1d
      case "Manual"         => 2d
      case "Management"     => 3d
      case "Skilled Manual" => 4d
    },
    model.region match {
      case "North America" => 0d
      case "Pacific"       => 1d
      case "Europe"        => 2d
    },
    model.totalChildren.toDouble,
    model.yearlyIncome)
}