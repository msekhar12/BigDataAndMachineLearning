## DATA622 HW #5
- Assigned on October 25, 2017
- Due on November 15, 2017 11:59 PM EST
- 15 points possible, worth 15% of your final grade

### Instructions:

Read the following:
- [Apache Spark Python 101](https://www.datacamp.com/community/tutorials/apache-spark-python)
- [Apache Spark for Machine Learning](https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning)

Optional Readings:
- [Paper on RDD](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf)
- [Advanced Analytics with Spark: Patterns for Learning from Data at Scale, 2nd Edition](https://www.amazon.com/_/dp/1491972955), Chapters 1 - 2

Additional Resources:
- [Good intro resource on PySpark](https://annefou.github.io/pyspark/slides/spark/#1)
- [Spark The Definitive Guide](https://github.com/databricks/Spark-The-Definitive-Guide)
- [Google Cloud Dataproc, Spark on GCP](https://codelabs.developers.google.com/codelabs/cloud-dataproc-starter/)


### Critical Thinking (8 points total)

1. (2 points) How is the current Spark's framework different from MapReduce?  What are the tradeoffs of better performance speed in Spark?
>> Spark uses in-memory framework to perform its computations. In Spark, the transformations are NOT evaluated immediately. Instead, all the transformations are recorded into a DAG (Directed Acyclic Graph), and as soon an action is requested, Spark optimizes the DAG, and evaluates the transformations to obtain the output for the requested action. In case of failure, Spark will recover the in-memory data, by performing the transformations and actions in optimized DAG.
>> In MapReduce the data between Map and Reduce phase are persisted to disk causing IO (Input/Output) operations. In case of MapReduce, to avoid re-computation (in case of failures), we can save the intermediate results to HDFS data sets, and continue the job by reading the last saved data.
>> In Spark, due to lazy evaluation, in some scenarios MapReduce might have an edge over Spark to recover from failures, as MapReduce will persist data between phases to disk.


2. (2 points) Explain the difference between Spark RDD and Spark DataFrame/Datasets.
>> RDDs help us to handle unstructured data, whereas the DataFrame/Datasets are meant for structured and semi-structured data. If we need more control with transformations and actions, then RDDs are best to use. Also if we do NOT have Spark 2 or later, then you have to use RDDs, as the DataFrame/Datasets are not available before Spark 2
>> In Spark DataFrame/Datasets the data is organized into rows and columns. We can treat Spark DataFrame as language specific constructs like Python DataFrames (using Pandas), R DataFrames etc. This makes the DataFrame as an easier construct to work with, when compared to RDD.
>> Spark DataSets need static typing (means you have to define the data types of the columns), wheras DataFrame is Un-typed. Spark Datasets are available only in Java and Scala. 
>> Spark DataFrames/Datasets internally use RDDs

3. (1 point) Explain the difference between SparkML and Mahout.  
>> Mahout was built on MapReduce framework, while SparkML was built on Spark RDD framework. Mahout is an older Machine Learning library, while SparkML is a relatively newer Machine Learning library. Since Mahout was built on MapReduce, which uses Disk I/O to persist the data between phases, Mahout is much slower in evaluation/optimization/convergence of Machine Learning algorithms. Spark uses RDD, which uses in memory computation to a maximum extent, and hence SparkML runs much faster than Mahout.  
>> As per wikipedia (https://en.wikipedia.org/wiki/Apache_Mahout), support for MapReduce based algorithms is gradually phased out of Apache Products. So the support for Mahout might be discontinued in future.

4. (1 point) Explain the difference between Spark.mllib and Spark.ml.
>> Spark.mllib was built on RDD, while Spark.ml was built based on the DataFrames concept. Hence Spark.ml supports Pipelines to apply transformations and training in a single construct.
>> Since Spark.ml supports Pipelines, it is advisable to use Spark.ml wherever possible. But not all algorithms are available in Spark.ml, as it is relatively new.   

4. (2 points) Explain the tradeoffs between using Scala vs PySpark.
>> Spark based applications developed using Scala give better performance than PySpark. All the brand new features are available in Scala first. 
>> In Scala the type checking happens at compile time, forcing the developer to fix the unnoticed bugs early. In Python (PySpark), type checking happens at run-time. So errors in PySpark applications will be unnoticed till the relevant code is executed.
>> Spark DataSets are statically typed, and hence they are not available in PySpark. But Scala being a statically typed language supports Spark DataSets. 

### Applied (7 points total)

Submit your Jupyter Notebook from following along with the code along in [Apache Spark for Machine Learning](https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning)
