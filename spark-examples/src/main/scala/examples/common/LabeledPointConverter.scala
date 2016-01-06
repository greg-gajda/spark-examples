package examples.common

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector

trait LabeledPointConverter {
  def label(): Double
  def features(): Vector
  def toLabeledPoint() = LabeledPoint(label(), features())
}