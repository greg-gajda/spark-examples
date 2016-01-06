# Classification of customers by using different algorithms

Here are notes from examining some classification algorithms available in Spark’s MLLib with data excerpted from AdventureWorksDW2012 sample database. 

Data is copied into bike-buyers.txt file, which you can find on github (https://github.com/grzegorzgajda/spark-examples). 

This file contains following information about customers of fictitious Adventure Works Cycles multinational manufacturing company:

* CustomerKey – internal key assigned to every customer 
* Age – customer’s age,
* BikeBuyer – flag indicating that customer was bicycle buyer,
* CommuteDistance – working day commute distance,
* EnglishEducation –level of education,
* Gender – customer’s gender,
* HouseOwnerFlag - whether customer owns a house or not,
* MaritalStatus - marital status,
* NumberCarsOwned - number of cars owned by customer,
* NumberChildrenAtHome – number of children on education,
* Occupation – customer’s occupation,
* Region – region of living,
* TotalChildren -  how many children customer has in general,
* YearlyIncome – customer’s salary.

Spark’s MLLib supervised classification algorithms use data structure called LabeledPoint, which is a local vector, dense in our case, associated with a label/response. 

In general, variables (and data) either represent measurements/values on some continuous scale, or represent information about some categorical or discrete characteristics. 

For example, incomes of customers represent continuous variable; however, a person's gender, commute distance, occupation, or marital status are categorical or discrete variables: either a person is male or female, single or married, etc. 

Some variables could be considered in either way. For example, a customer's age may be considered a continuous variable, or discrete variable with finite categories, which size can be calculated from data provided. 

All examples are written in Scala, and most of them has also Java 8 versions. 

They can be run in Local or in Standalone Cluster mode. Customer’s data can be loaded from many different sources. 

Here, following options are taken into account:
* Local file,
* HDFS,
* Cassandra.

To build project Gradle is used.