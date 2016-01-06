# Sample calculations

Suppose after random split, test data set contains 1774 samples, 875 positive and 899 negative. 
If trained classifier predicts 855 samples as true with 751 actually true, and predicted 919 as false with 795 being false, then Confusion matrix will be:

|			         |Predicted as negative	|Predicted as positive|
| ------------------ |:------------:| ----------------:|
|Labelled as negative| 	795.0  	(TN)|		104.0  (FP)|
|Labelled as positive|	124.0  	(FN)|		751.0  (TP)|


Overall accuracy is (751 + 795)/(875 + 899) = 1546/1774 = 0.871, 

and misclassification for class 0 and 1 is 104/899 = 0,115 and 124/875 = 0,145 accordingly. 

For positive class Precision equals to 751/855 = 0,8783 and Recall equals to 751/875 = 0,8582. 

MCC is equal to 0,7430.

It can be expected, that for different numbers of examples per class, the average per-class accuracy will be different from the overall accuracy. 

Bike buyers data set is well balanced, it contains 9132 positive and 9352 negative examples, so the accuracy separately calculated for each class is 795/899 = 0,884 and 751/875 = 0,858, and average value 0,871 is exactly the same as overall accuracy.

If the classes were not balanced, then the accuracy would give a very distorted picture, because the class with more examples will dominate the statistic and both the average and the individual per-class accuracy should be checked. 

Now suppose data set contains 1575 positive and 199 negative samples. 
If classifier predicted 1451 as true with 1555 actual true, and 95 as false with 119 actual false, then Confusion matrix will be:

|			         |Predicted as negative	|Predicted as positive|
| ------------------ |:------------:| ----------------:|
|Labeled as negative |	95.0  (TN)  |		104.0  (FP)|
|Labeled as positive |	24.0  (FN)  |	    1451.0 (TP)|

Overall accuracy is (1451 + 95)/(1575 + 199) = 1546/1774 = 0.8714, 
and misclassification for class 0 and 1 is 104/199 = 0,5226 and 24/1575 = 0,0152 accordingly. 

The accuracy calculated for each class is 95/199 = 0,4773 and 1451/1575 = 0,9212 and average of them is 0,6956. 

For positive class Precision equals to 1451/1555 = 0,9331 and Recall equals to 1451/1475 = 0,9837, 
but for negative class Precision equals to 95/119 = 0,7983 and Recall equals to 95/199 = 0,4773. 

In this case MCC is equal to 0,5808.
