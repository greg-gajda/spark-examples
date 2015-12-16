A classification ensemble model is the combination of two or more classification models.
Random forest is an ensemble tree technique that builds a model with defined number of decision trees 
that is better classifier than average tree in the forest. So the whole performs better than any of its parts.

In Spark Random Forest model is created on decision tree algorithm with default voting strategy, 
where each tree votes on classification, and higher number of votes wins.

According to Spark documentation Random forests are one of the most successful machine learning models for classification and regression,
with reduced risk of overfitting. 
The algorithm injects randomness into the training process so that each decision tree is a bit different.
