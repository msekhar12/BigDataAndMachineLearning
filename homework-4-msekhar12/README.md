## DATA622 HW #4
- Assigned on October 11, 2017
- Due on October 25, 2017 11:59 PM EST
- 15 points possible, worth 15% of your final grade

### Instructions:

Use the two resources below to complete both the critical thinking and applied parts of this assignment.

1. Listen to all the lectures in Udacity's [Intro to Hadoop and Mapreduce](https://www.udacity.com/course/intro-to-hadoop-and-mapreduce--ud617) course.  

2. Read [Hadoop A Definitive Guide Edition 4]( http://javaarm.com/file/apache/Hadoop/books/Hadoop-The.Definitive.Guide_4.edition_a_Tom.White_April-2015.pdf), Part I Chapters 1 - 3.

### Critical Thinking (10 points total)

Submit your answers by modifying this README.md file.

1. (1 points) What is Hadoop 1's single point of failure and why is this critical?  How is this alleviated in Hadoop 2?
>> In Hadoop 1, NameNode was considered as a single point of failure, since there was only one instance of NameNode running in the cluster, and the entire Hadoop system could be useless if the data on NameNode is corrupted. As the NameNode hosts the metadata ("blocks" information) of all the files stored on HDFS, NameNode is very critical. If the disk on NameNode is damaged, then the meta data of all files stored on HDFS will be lost making the entire Hadoop environment to fail.
>> In Hadoop 2, we can have 2 NameNodes running (active and standby NameNodes). If the active NameNode fails, the standby NameNode will take over the responsibility of active NameNode, and hence the chances of losing data stored on HDFS in Hadoop 2 is very less.
>> In Hadoop 1 the NameNode's data is copied to a remote NFS file system, so that its data can be recovered, if the NameNode fails. This is still a predominant method in Hadoop 2 also, as a Disaster Recovery mechanism. Also NameNode's failure can be minimized if it is hosted on high end machine (instead of the commodity hardware).

2. (2 points) What happens when a data node fails?
>> Hadoop, maintains 3 copies of file's blocks by default. Each copy is maintained on separate machines (2 blocks on different machines belonging to the same rack, and one block on a machine in a different rack).
>> So whenever a DataNode fails, then the file blocks on that DataNode are automatically replicated to another available DataNode, so that we will always have 3 copies of the blocks.
>> The data belonging to the failed DataNode can be accessed from other DataNodes (where the failed DataNode's contents are replicated).

3. (1 point) What is a daemon?  Describe the role task trackers and job trackers play in the Hadoop environment.
>> A daemon is a background process that can run without any user control. Most of the started tasks in Unix/Linux environments are daemons, since these tasks are started even before any user logs into the server, and continue to run irrespective of any user activity.
>> When we run a MapReduce job, we submit the job through JobTracker. JobTracker is a daemon, which splits the input job into Mappers and Reducers. The Mappers and Reducers will run on other machines of the cluster. Running the actual Mappers and Reducers is handled by another daemon called TaskTracker. The TaskTracker will run on each machine, wherever the DataNode daemon runs.
>> Since the TaskTracker and DataNode runs on the same machine the Hadoop framework will help the Map task to run directly on the data stored on the same machine. It will avoid a lot of network traffic. 

4. (1 point) Why is Cloudera's VM considered pseudo distributed computing?  How is it different from a true Hadoop cluster computing?
>> Cloudera's VM will be run as a virtual environment of VM Ware (or Oracle VM Virtual Box). It mimics that we have a cluster of machines, although we are running the virtual environment on our laptop or desktop. Hence it is called "pseudo distributed" environment.
>> In a true Hadoop cluster we have NameNode and Job trackers running on separate (or same) machines, secondary NameNode on a different (most of the time on another high end machine), and the DataNodes and TaskTrackers running on commodity hardware or mid range or high end servers.
>> So in real Hadoop environment we will have a cluster of machines, which will be used to store (using HDFS) and process (using MapReduce) huge volumes of data.

5. (1 point) What is Hadoop streaming? What is the Hadoop Ecosystem?
>> Hadoop streaming is a feature of Hadoop that mimics the streaming of data as if the data is received and sent from/to the standard input device (keyboard)/output device (monitor).
>> Using Hadoop streaming we can write MapReduce programs in any language (which supports the processing of standard input and output data). 
>> Hadoop eco system consists of all the core components of Hadoop (HDFS and MapReduce), along with the other utility software packages developed to utilize Hadoop for big data processing. 

6. (1 point) During a reducer job, why do we need to know the current key, current value, previous key, and cumulative value, but NOT the previous value?
>> We need to know the current key, since we can compare the current key value with the previous key value to determine if the key has changed or not.
>> We need the current value, since we can add the current value to the cumulative value, if the previous key and the current key are same. If the current key and previous key are not same, we have to write the (key,cumulative value) pair to the standard output (indirectly to HDFS), followed by the initialization of the cumulative value to the current value, and previous key to the current key.
>> We need the previous key, since we have to compare the current key with the previoud key to determine whether to add the current value to the cumulative value or to write the previous key and cumulative value to HDFS.
>> We need the cumulative value, so that we can keep track of the sum of the values for a given key.
>> We do NOT need the previous value, since its value is already factored into the cumulative value.

7. (3 points) A large international company wants to use Hadoop MapReduce to calculate the # of sales by location by day.  The logs data has one entry per location per day per sale.  Describe how MapReduce will work in this scenario, using key words like: intermediate records, shuffle and sort, mappers, reducers, sort, key/value, task tracker, job tracker.  
>> Background: While storing the data into HDFS, the data is divided into blocks, and distributed across the data nodes. Whenever we submit a MapReduce job, the job tracker will divide the job into mappers and reducers, and send the mappers to the data nodes, where the "to be processed" file's blocks are existing. On the DataNode, the TaskTracker will run the Mappers. 
>> Task Tracker will run a single mapper per block of the file. So for each block, we will have one mapper. In rare circumstances, the TaskTrackers (where the target data is residing) might be busy. In such cases, the file's blocks are streamed to a different DataNode, so that the TaskTracker on that DataNode will process the data.

>> Given that we have a file with sale location and date of sale. Our goal is to find the number of sales for each location for each day. In other words, we have to find the counts, grouped by the location and date.
>> We have to use date and store location as the key. 
>> In the mapper, we will loop over the contents of the HDFS file (by pretending that the data is streaming from standard input, if the program is written in python or non-java language). We will extract the store location and date. The location and date will be the key (composite key), and we will assign 1 as value for each record. The key value pairs are written as a csv file or tab delimited value to the standard output.
>> The (key,value) pairs from each mapper are shuffled and sorted, so that the data from shuffle and sort phase will be in the order of locations, and date (composite key).
>> The data from shuffle and sort phase will be sent to the reducer(s).Hadoop will make sure that each reducer will work on data related to a single key, and a reducer can process more than one key's data. 
>> We know that the reducer receives data sorted by location and date (as they form the key).
>> In the reducer, we will create a counter and initialize it to 0. For each (key, value) pair, we will increase the counter, if the previous key is same as the current key value. So if the current key (location, date) is same as  the previous key (location, date), then increment the counter. If the keys are different, then write the key (location, date) and counter as a tab delimited or comma delimited value to the standard output, and the counter is reset to 0.
>> At the end of the loop, you have to check if the counter is NOT 0, and write the location, date, and counter to the standard output.


### Applied (5 points total)

Submit the mapper.py and reducer.py and the output file (.csv or .txt) for the first question in lesson 6 for Udacity.  (The one labelled "Quiz: Sales per Category")  Instructions for how to get set up is inside the Udacity lectures.  
