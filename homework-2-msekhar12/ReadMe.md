## DATA622 HW #2
- Assigned on September 13, 2017
- Due on September 27, 2017 11:59 PM EST
- 15 points possible, worth 15% of your final grade


### Data Pipeline using Python (13 points total)

Build a data pipeline in Python that downloads data using the urls given below, trains a random forest model on the training dataset using sklearn and scores the model on the test dataset.

#### Scoring Rubric
The homework will be scored based on code efficiency (hint: use functions, not stream of consciousness coding), code cleaniless, code reproducibility, and critical thinking (hint: commenting lets me know what you are thinking!)  

#### Instructions:
tl;dr: Submit the following 5 items on github.  
- ReadMe.md (see "Critical Thinking")
- requirements.txt
- pull_data.py
- train_model.py
- score_model.py

More details:

- <b> requirements.txt </b> (1 point)

This file documents all dependencies needed on top of the existing packages in the Docker Dataquest image from HW1.  When called upon using <i> pip install -r requirements.txt </i>, this will install all python packages needed to run the .py files.  (hint: use pip freeze to generate the .txt file)

- <b> pull_data.py </b> (5 points)

When this is called using <i> python pull_data.py </i> in the command line, this will go to the 2 Kaggle urls provided below, authenticate using your own Kaggle sign on, pull the two datasets, and save as .csv files in the current local directory.  The authentication login details (aka secrets) need to be in a hidden folder (hint: use .gitignore).  There must be a data check step to ensure the data has been pulled correctly and clear commenting and documentation for each step inside the .py file.

    Training dataset url: https://www.kaggle.com/c/titanic/download/train.csv
    Scoring dataset url: https://www.kaggle.com/c/titanic/download/test.csv

- <b> train_model.py </b> (5 points)

When this is called using <i> python train_model.py </i> in the command line, this will take in the training dataset csv, perform the necessary data cleaning and imputation, and fit a random forest classifier to the dependent Y.  There must be data check steps and clear commenting for each step inside the .py file.  The output for running this file is the random forest model saved as a .pkl file in the local directory

- <b> eda.ipynb </b> (0 points)

[Optional] This supplements the commenting inside train_model.py.  This is the place to provide scratch work and plots to convince me why you did certain data imputations and manipulations inside the train_model.py file.

- <b> score_model.py </b> (2 points)

When this is called using <i> python score_model.py </i> in the command line, this will ingest the .pkl random forest file and apply the model to the locally saved scoring dataset csv.  There must be data check steps and clear commenting for each step inside the .py file.  The output for running this file is a csv file with the predicted score, as well as a png or text file output that contains the model accuracy report.  


### Critical Thinking (2 points total)

Modify this ReadMe file to answer the following questions directly in place.

1. If I had to join another data source with the training and testing/scoring datasets, what additional data validation steps would I need for the join? (0.5 points)
   >>> Let us assume that we have additional information about the country of citizenship of the passengers, and we would like to join this data to the training/testing data. To perform this, make sure that the country names are properly spelled. For example check if different case is used (example: Australia,australia,AUSTRALIA must belong toe the same country, although it is spelled in different case).
       So it is advisable to convert the country name to the same case (all capitals or small letters). Then make sure that the incoming data has the proper join key PassengerId. If there is NO passenger ID, check if the data can be joined by some other key or find how to assign the passengerId to the incoming data.
       Make sure that you use left outer join between existing data and new data, since we do not know if country of citizenship is available for all the passengers. Then check how many levels are present for the country of citizenship, and if there are not many levels, then create dummy variables to represent the country of citizenship. However, if there are many levels (more than 10 countries), then bin the data as per frequency distribution.  	   
	   
  

2. What are some things that could go wrong with our current pipeline that we are not able to address?  For your reference, read [this](https://snowplowanalytics.com/blog/2016/01/07/we-need-to-talk-about-bad-data-architecting-data-pipelines-for-data-quality/) for inspiration. (0.5 points)
  >>> Extrapolation or applying the model using data falling outside the range of the training data leads to incorrect predictions. We have to validate the incoming data and place automatic alerts, if we are extrapolating the model. 
      We are not converting all the text data to the same case, so our text is case sensitive, and this can result in discarding important data. For instance we created a dummy variable for Sex = 'male'. But if the input data has Sex = 'Male', then we our pipeline will simply consider the value as not 'male', resulting in inaccurate predictions.
	  Since the null values handling logic has been embedded in the pipelines, our model should not break, irrespective of the presence of null values in any incoming/new data.

3. How would you build things differently if this dataset was 1 trillion rows? (0.5 points)
   >>> It depends. If we have distributed computing environment such as Hadoop, then we can use Spark's MLLlib to build our models. If we do not have distributed computing environment, then I divide the data into small chunks (perhaps 1 million records or just 0.5 million records, based on the RAM size), and fit a model considering each chunk as a separate training data.
       We will get many models, and we can combine the outputs of all these models (by getting average of all predictions for regression problems or getting the majority votes for classification).

4. How would you build things differently if the testing/scoring dataset was refreshed on a daily frequency? (0.5 points)
   >>> Validation: We know that we achieved a test accuracy of 0.83 in our model development. Along with this score we also have the accuracy scores obtained in Cross Validation. Using these scores, we can get the average accuracy and the standard deviation of the accuracies. 
       Now for new data, we will use our model to predict the target label, and compare the prediction with the actual label (which will be captured later, once the actual result is known). If the accuracy is within 1.5 times the standard deviation from the median accuracy, then our model is predicting well.
	   We will save this accuracy in our list of accuracies. 
   >>> If the accuracy is abnormal (something like less than 0.5 or too good to be true accuracy like 0.98), then we need get alerted and check the data manually, and see if there is any abnormal data or if we are getting the same data we used for training (resulting is accuracy of almost 1).	   
	   
   >>> Updating model (if there is lot of variation in the incoming data):
	   Once we accumulate considerable amount of data, we can develop a new model, and combine the predictions of the new model and old model(s) to predict the new data. 
       For example, if we developed M1, M2, M3 ... M100 models(M100 being the latest), then we can use a decreasing set of weights to get the final prediction (0.99 * M100 + 0.98 * M99 ... 0.01*M1). Such model will give more importance to the newer models than the older models.
	   We can also reverse the weights, to give more emphasis on the older model and less importance to the newer models. In any case the strategy to update the model must be thoroughly tested before implementing the change in production.