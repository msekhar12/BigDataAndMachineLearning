## DATA622 HW #3
- Assigned on September 27, 2017
- Due on October 11, 2017 11:59 PM EST
- 15 points possible, worth 15% of your final grade

### Instructions:
1. Get set up with free academic credits on both Google Cloud Platform (GCP) and Amazon Web Services (AWS).  Instructions has been sent in 2 separate emails to your CUNY inbox.

2. Research the products that AWS and GCP offer for storage, computing and analytics.  A few good starting points are:
    - [GCP product list](https://cloud.google.com/products/)
    - [AWS product list](https://aws.amazon.com/products)
    - [GCP quick-start tutorials](https://codelabs.developers.google.com/codelabs)
    - [AWS quick-start tutorials](https://aws.amazon.com/getting-started/tutorials/)
    - [Mapping GCP to AWS products Azure](https://stackify.com/microsoft-azure-vs-amazon-web-services-vs-google-compute-comparison/)
    - [Evaluating GCP against AWS](http://blog.armory.io/choosing-between-aws-gcp-and-azure/)


3. Design 2 different ways of migrating homework 2's outputs to *GCP*.  Evaluate the strength and weakness of each method using the Agile Data Science framework.

4. Design 2 different ways of migrating homework 2's outputs to *AWS*.  Evaluate the strength and weakness of each method using the Agile Data Science framework.

### Critical Thinking (8 points total)

- Fill out the critical thinking section by modifying this README.md file.
- If you want to illustrate using diagrams, check out [draw.io](https://www.draw.io/), which has a nice integration with github.

**AWS Method 1** (2 points)

Server Migration Services

Description:
>>If we want to migrate virtual servers to AWS, Server Migration Services is one of the best methods, as it has minimum downtime, and we can concurrently migrate 50 virtual servers at a time. 
>>This service is mainly targetted at VM Ware virtual machines. 

Strengths:
>>Can migrate virtual server environments easily and almost seamlessly to AWS. It is cost effective, as this Migration service is free to use, and we need to pay only for storage resources used during migration.
>>So if we have developed a complete machine learning application in VM Ware virtual environments, then we can utilize Server Migration Services to migrate our complete application, including the OS environment directly to AWS.
>>It is suitable for agile data science framework, since we can measure the number of servers we migrated in each unit of time (like a sprint), and provide some deliverable (like the number of servers migrated) at the end of the sprint.
>>We can plan the order of migration of local virtual machines based on the applications dependency, and hence we can measure the number of applications migrated in each sprint. 
>>Therefore this migration method supports agile data science framework (delivering a workable piece of code or atleast some experiment results at the end of each sprint)

Weaknesses:
>>It is mainly targetted for VM Ware virtual servers only. We cannot use this service to migrate the applications which are developed in the physical servers or other hypervisors, although such support is expected to come soon.
>>It is not available at all the regions.
>>Once the replication process is initiated, we have 90 days to finish the migration per server. 
>>If we are in an area where high bandwidth for internet is not available, and if we need to transfer a huge volume of data, then this might not be an optimal method. 
>>Available only for Windows, and some Linux based environments. 

**AWS Method 2** (2 points)

Snowball Edge

Description:
>>Snowball Edge is a 100TB data transfer device with on-board storage and compute power for select AWS capabilities. 
>>In addition to transferring data to AWS, Snowball Edge can undertake local processing and edge-computing workloads. 
>>Features include an S3-compatible end-point on the device, a file interface with NFS support, a cluster mode where multiple Snowball Edge devices can act as a single, scalable, storage pool with increased durability.

Strengths:
>>We can migrate huge volumes of data easily. This method is also suitable for agile data science frame work, since at the end of each sprint, we can provide some deliverable (such as the amount of data copied to the snowball edge)

Weaknesses:
>>It is not free. Each Snowball Edge job costs a flat fee of $300 for device handling and operations at AWS data centers. The $300 job fee includes 10 day of use at on-site. 
>> Beyond that, a Snowball Edge costs $30/day for each extra day that it is at the site.
>>If our application is an online application, then the data needs to be synced up after the snowball is shipped and copied to AWS. 


**Other simple methods**
If we have a set of small files to be copied, we can use Win SCP (for windows environment) or SCP (Secure Copy) for all other environments. Other methods can be FTP (File Transfer Protocol), if enabled on the AWS instance. 

**GCP Method 1** (2 points)
>>VM Migration service. 

Description:
>>In GCP we can migrate Virtual Machines using VM Migration service. 

Strengths:
>>Free to use. 
>>As described for AWS method 1, this migration method is suitable for agile data science framework, since we can measure the number of servers we migrated in each unit of time (like a sprint), and provide some deliverable at the end of the sprint (something like the number of virtual servers migrated to GCP).

Weaknesses:
>>To use the VM migration service, the source machines from which you are migrating must be running one of the following operating systems:

>>Microsoft Windows Server 2008 R2 64 bit (Datacenter Edition), Microsoft Windows Server 2012 R2 64 bit (Datacenter Edition), Microsoft Windows Server 2016 64 bit (Datacenter Edition)

>>SUSE Linux (SLES) 11 or above, Debian Linux 8, Kali Linux 2.0, Ubuntu 12.04 or above, Red Hat Enterprise Linux (RHEL) 5.0 or above, CentOS 6.0 or above, Oracle Linux 6.0 or above



**GCP Method 2** (2 points)
>>Transferring files through Cloud Storage

Description:
>>You can use Cloud Storage buckets to transfer files to and from your instances. This file transfer method works on almost all operating systems and instance types as long as your instance has access to your Cloud Storage bucket through a service account or through your personal user credentials.

>>Upload your files from your workstation to a Cloud Storage bucket. Then, download those files from the bucket to your instances.

Strengths:
>>The transfer of files in either direction is possible.
>>This method is also suitable for agile data science framework, since we can measure the number of files transferred or the amount of file transferred to cloud bucket, and hence can provide some deliverable (like the amount of data copied or files copied in the sprint)

>>There is no limitation of the sizes of the files. As long as you have the bandwidth to upload to storage bucket, you are good.

Weaknesses:
>>Cloud storage is object based storage. This means, you cannot copy (or install) your software installations to cloud storage, and download from cloud storage to cloud instance.
 
**Other simple methods**
>>The easiest method to copy files to google cloud virtual machines is using SCP or gcloud copy. gcloud copy can easily copy the files to/from local machine with a simple copy-files command.

>>For question 7, I used cloud shell, and used "upload file" option (see the provided screen shots for more details).


### Applied (7 points total)

Choose one of the methods described above, and implement it using your work from homework 2.  Submit screenshots in the *screenshot* folder on this repo to document the completion of your process.
