package examples.features


import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.SparkConf
import org.apache.spark.annotation.Experimental
import org.apache.spark.annotation.Since

object TFIDFOnWiki {

  case class Person(url: String, name: String, info: String)

  def wordsCount(words: Array[String]) = {
    words.foldLeft(Map.empty[String, Int])((m, s) => m + (s -> (1 + m.getOrElse(s, 0))))
  }

  def TfIdf(rdd: RDD[Person]) = {
    //instead of guessing numFeatures, take rather distinct words count 
    val tf = new HashingTF(numFeatures = 100000)
    val data = rdd.map { p => (p.name, tf.transform(p.info.split(" "))) }
    val vectors = data.map(_._2).cache

    val idfModel = new IDF().fit(vectors)

    val idf = idfModel.transform(vectors)

    val tfidf = idf.zip(data).map(f => (f._2._1, (f._2._2, f._1)))
    tfidf
  }

  def main(args: Array[String]): Unit = {
    
    val conf = new SparkConf().setAppName("Features Extraction TF-IDF").setMaster("local[8]")
    val sc = new SparkContext(conf)
    val wFile = sc.textFile("data/wiki_people.csv")
    val people = wFile.map(_.split(";")).map(a => Person(a(0), a(1), a(2)))

    val sparkTFIDF = TfIdf(people)

    //terms frequencies within document
    val tfs = people.map(p => (p.name, wordsCount(p.info.split(" "))))
    val docs = tfs.count.toDouble

    //terms frequencies across documents
    val dfs = tfs.flatMap(_._2.keySet).map((_, 1)).reduceByKey(_ + _)
    val idfs = dfs.map {
      case (term, count) => (term, math.log(docs / (1 + count)))
    }.collectAsMap

    //self-calculated TF-IDF
    val selfTFIDF = tfs.map { case (name, wc) =>
      (name, wc.map {
        case (word, count) => (word, count * idfs.getOrElse(word, 0d))
      })
    }

    //sparkTFIDF.sortBy(f => f._2._1, ascending=false)

    selfTFIDF.take(10).foreach(println)

    //Al Pacino
    //Lady Gaga
    //Ryan Gosling
    //Marina Kulik -dutch painter

    //turn terms frequencies across documents into indexes and values
    val zidfs = idfs.zipWithIndex.map(mi => (mi._1._1, mi._2))

    sc.stop
  }
}