# Decision Trees Tuning 

Spark decision tree classification algorithm takes some parameters that can significantly influence quality of created model. 
There is no simple theory that tells what values should be used. Answer can be found by iterations, and selecting model with best characteristics. 
```scala
    val numClasses = 2
    val tuning =
      for (
        impurity <- Array("entropy", "gini");
        maxDepth <- Range(5, 25, 5);
        maxBins <- Range(10, 50, 2)
      ) yield {
        val model = DecisionTree.trainClassifier(train, numClasses, BikeBuyerModel.categoricalFeaturesInfo(), impurity, maxDepth, maxBins)
        val predictionsAndLabels = test.map {
          point => (model.predict(point.features), point.label)
        }
        val stats = Stats(confusionMatrix(predictionsAndLabels))
        val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
        val auPR = metrics.areaUnderPR()
        val auROC = metrics.areaUnderROC()        
        ((impurity, maxDepth, maxBins), stats.MCC, stats.ACC, auPR, auROC)
      }
    tuning.sortBy(_._2).reverse.foreach{
      x => println(x._1 + " " + x._2 + " " + x._3+ " " + x._3)
    }
```
Example output can look like that:
(gini,20,32) 0.7591468068014129 0.8795698924731182 0.9094022702677811, 0.8795698924731183
(gini,20,36) 0.7561452549554005 0.8779569892473118 0.9069982737027384 0.8779569892473119
(entropy,20,20) 0.7527160219196528 0.8763440860215054 0.9078155249224453 0.8763440860215054
(entropy,20,28) 0.7494662650097887 0.874731182795699 0.9058491511945511 0.874731182795699
(gini,20,28) 0.7473122599794635 0.8736559139784946 0.9053086334016228 0.8736559139784947
(gini,20,48) 0.7463211280086185 0.8731182795698925 0.9058083298117106 0.8731182795698925
…

Results is sorted by MCC, but it can be noticed that it is also almost perfectly sorted by accuracy and values of areas under ROC and PR curves. Now better decision tree parameters can be used to create better classification model.
