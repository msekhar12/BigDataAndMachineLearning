#################################
##Program Name: train_model.py ##
##Python version: 3.4          ##
##Author: Sekhar Mekala        ##
#################################

##Importing all the required packages:

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer, Imputer, MinMaxScaler
from sklearn.pipeline import FeatureUnion

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict

from sklearn.externals import joblib

##############################################################
##Reading the file train.csv to a pandas data frame called df#
##############################################################
def read_file(file_name):

    #Always supply the data types details,
    #as pandas will read the input file in chunks,
    #and if dtype option is not supplied, then pandas will
    #just infer the data type of the columns, and 
    #if the data type of a column is interpreted as float, as it has only
    #numeric values in the first chunk, but in the second chunk has a str value,
    #then read_csv() will fail in reading the second file chunk.
    
    dtype = {'PassengerId':'int', 
    'Survived':'int',
    'Pclass':'str',
    'Name':'str',
	'Sex':'str',
    'Age':'float',
    'SibSp':'int',
    'Parch':'int',
    'Ticket':'str',
    'Fare':'float',
    'Cabin':'str',
    'Embarked':'str'
    }
    
    df = pd.read_csv(file_name,dtype=dtype)
    try:
        df = pd.read_csv(file_name,dtype=dtype)
    except:
        print("EXCEPTION/ERROR in read_file() function. Terminating the program.")
        exit(10)        
    return df
        

def get_train_data(df):
    '''
    ########################
    ##Test Train split:   ##
    ########################
    Splitting the data into test (20%) and train(80%)
    The function will return a list of data frames (X_train,y_train), containing the source and target variables
    The function will save the test data as test_data.csv
    
    The file test_data.csv will be used by score_model.py    

    We will divide the Age variable into 8 ranges (Age category), so that we can stratify
    the test and train split based on the Age category.
    See the EDA document for the detailed analysis, for the reason behind this.
    '''
    
    #Create a new variable called 'age_cat'.
    #This variable will contain the age category, and will be used
    #to stratify the sampling, while performing test/train split
    #The goal is to have right mix of age data in both test and training data
    
    #Check if the input is a data frame. If not raise an exception and terminate the program.
    try:
        if not isinstance(df,pd.DataFrame):
           raise ValueError
    except:
        print("**EXCEPTION/ERROR**: get_train_data(). The input to this function MUST be a pandas data frame. Terminating the program.")
        exit(10) #Make a return code of 10, to handle errors, should we use this in a job/shell script
    
    try:    
        df['age_cat'] = np.ceil(df["Age"]/8)
    except:
        print("**EXCEPTION/ERROR**: get_train_data(). The input data frame does not have Age variable. Terminating the program.")
        exit(10) #Make a return code of 10, to handle errors, should we use this in a job/shell script
    
    #Create the age_cat column
    df["age_cat"].where(df["age_cat"] < 8, 8.0,inplace=True)



    ##We will use random seed of 42 (random_state=42) to reproduce the same results.
    ##We will get 4 data frames:
    ##X_train: This contains all the training data
    ##y_train: This contains the target class for the observations in X_train
    ##X_test: This contains all the test data
    ##y_test: This contains the target class for the observations in X_test
    split =  StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)
    X = df.drop(['Survived'],axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=X['age_cat'])

    ##Since age_cat is NOT needed anymore, we will drop it from all the X_train and X_test data frames:
    X_train.drop(['age_cat'],axis=1,inplace=True)
    X_test.drop(['age_cat'],axis=1,inplace=True)

    ##The X_test and y_test data frames will NOT be used in training.
    ##They will be used to evaluate our final model
    ##See program score_model.py
    ##Let us save the X_test and y_test data frames, since they will be used by 
    ##the program: score_model.py
    ##Combine the X_test and y_test data frames and save as test_data.csv
    try:
        X_test["Survived"] = y_test
        X_test.to_csv("test_data.csv",index=False)
    except:
        print("EXCEPTION/ERROR in get_train_data(). Error occurred while saving the test files.")

    print("Saved the test data to test_data_inputs.files and test_data_output.csv files")
    return (X_train,y_train)

#######################
##Building Pipelines: #
#######################

## We will use the following transformations. 
## Each these possible transformations might have different effect on Machine Learning algorithms performance.
## The below listed combinations are not exhaustive. The goal is to build a set of transformations,
## so that their effect can be tested using Cross Validation (CV) technique. In other words, we are creating our own 
## additional hyper parameters to tune the algorithms. For example we can create a hyper parameter to represent 
## the inclusion or exclusion of a column in tuning the algorithm using CV method.


## The Cabin column has a lot of Null values. So we will test two possible combinations: 
##  1. Drop the Cabin column
##  2. Replace the Cabin column with another column Cabin_Indicator. This new column will have 1 whenever 
##     the Cabin has a non-null value. If the Cabin variable has null vaue, then the new column will have a 0

## The Sex variable does not have any null value. We will use LabelBinarizer() to transform the values in Sex
## based on one-hot encoding method.


## The Pclass has 3 values 3, 2, 1. We will treat these as character variables 
## and apply LabelBinarizer() on this column also


##  The Age column has null values. So we will try the following 3 combinations:
##  1. Replace all null values with median
##  2. Replace all null values with mean
##  3. Replace all null values with most frequent value

## To all numeric columns, we have to two options to scale the data:
## Std scaler
## Min-Max scaler

## The _Embarked_ column has 22% of null values (or 2 rows. See EDA). We will test the following 2 options:
## 1. Treat the null value as a separate class called Embarked_Null, and apply LabelBinarizer().
##    Embarked_Null will have a 1, if the corresponding value in Embarked has a non-null value
##    else it will have a 0 value
   
##Pipelines helps us to evaluate the above transformations, and test, if those transformations help (or does not help) the models.
##We will use 5 fold Cross Validation technique to evaluate the model performance


## In the following code we inherit BaseEstimator and TransformerMixin classes
## into our customized classes (which will be used in building the pipelines later).
## BaseEstimator will give us fit() and transform() methods, while 
## TransformerMixi will give us fit_transform()

##Each class we defined below will have a local parameter called column_names.
##This variable will contain the actual column names of the data frame obtained after performing the transformation
##This will help us to track the column names, even though we get the numpy arrays in the Pipelines.

## The DataFrameSelector will help us to select the desired columns of a data frame
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
        self.column_names = attribute_names
    def fit(self,X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names]



##Cabin transformer: 
##Cabin is a categorical variable. But unlike other categorical variables, 
##we want to test the effect of the following:
##The Cabin column has many classes, and mostly Null values (77%). So is this variable really useful?
##We will test the models CV score by ignoring cabin column (or)
##We will create another column called Cabin_Indicator, which contains 1 
##if the corresponding value in Cabin is not null,
##else Cabin_Indicator will have 0.

class CabinTransformer(BaseEstimator, TransformerMixin):
    #The inclusion/exclusion of Cabin variable is controlled by strategy variable
    #Default strategy is 'drop'. Other possible value is 'binomial'
    #The parameter strategy will be a custom hyper parameter
    
    def __init__(self, strategy='drop'): 
        self.strategy = strategy
        self.column_names = []
    
    def fit(self, X, y=None):
        self.column_names = []
        return self #Nothing else to do
    
    
    def transform(self, X, y=None):
        ##If strategy is 'drop', then delete the Cabin column
        if self.strategy == 'drop':
            return X.drop(['Cabin'],axis=1)

        ##If strategy is 'binomial', then create Cabin_Indicator            
        if self.strategy == 'binomial':
            X.loc[:,('Cabin_Indicator')] = X['Cabin'].apply(lambda x: 1 if not pd.isnull(x) else 0)
            self.column_names = ['Cabin_Indicator']
            return X.drop(['Cabin'],axis=1)

            
            
            
##The following transformation will be applied to all categorical variables, except Cabin, 
##since Cabin is handled by CabinTransformer
##CatNullTransformer will replace all the NaN values in categorical columns with a parameter value supplied as input.
##The value is supplied as input to the class constructor. 
class CatNullTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,value): 
        self.value = value
        self.column_names = []
    
    def fit(self, X, y=None):
        return self #Nothing else to do
    
    def transform(self, X, y=None):
            self.columns = list(X.columns)
            for i in self.columns:
                X.loc[:,(i)] = X[i].fillna(self.value)
            return X


#The following transformer will be  applied to all categorical variables except Cabin (as it is handled separately)
#The below code might look complex, but I had to write this logic due to the following drawbacks
#of the existing encoders:
##1. We can use get_dummies of pandas. 
##   But without modifying or creating a custom transformer, it is not possible to use it in pipeline.
##   For example, if we have ['r','g'] in column color, then get_dummies() will give us following:
##   color_r color_g
##   ------- -------
##   1       0
##   0       1
##   But if our test data (future data)  has an extra class 'b' or does not have 'r' or 'g', then we will bet different dummy variables.


##2. We can use LabelBinarizer(). But it has the following drabacks:
##2a. LabelBinarizer cannot take more than one categorical column as input.
##2b. LabelBinarizer behaves differently based on the number of unique classes in the input categorical variable.
##    For example, if we fit LabelBinarizer on a categorical variable with values ['red','blue'], then it will create
##    a model of LabelBinarizer. Now if we aply transform() on ['red','blue'], then we will get a one column variable
##    [1, 0]. However if we try to transform ['red','blue','green'], using the same LabelBinarizer model fit on ['red','blue'], then
##    we will get a matrix (like the following):
##    [[0,1],
##     [1,0],
##     [0,0]]
##    The shape of the matrix changes if transform() is applied on more than 2 levels. 
##    Ideally we should get the following transformation when applied on ['red','blue']
##    [[1,0],
##      [0,1]]
##    So that any new level (like 'green') will be represented as [[0,0]]

##
##2c. Another drawback of LabelBinarizer is, if we have only 2 classes in a variable, then only one dummy variable is created to 
##    represent the data. But if we have more than 2 classes, then the number of dummy variables is equal to the number of
##    classes. This discrepancy will cause confusion if the transformation is combined in pipelines. 

##Hence I had to create the following class...
##The following transformer must be applied on pandas data frame only. This transformer can process multiple categorical
##columns and also creates dummy variables based on the principle: n dummy variables per categorical variable, if the 
##categorical variable has n classes. 
            
class CatMultiLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self): 
        pass ##Nothing else to do
    
    def check_input_obj(self,X,location):
        ##Check if input object is a pandas df, else raise exception
        try:
            if not isinstance(X,pd.DataFrame):
               raise ValueError
        except:
            print("**EXCEPTION/ERROR**: In "+ location + " function of CatMultiLabelTransformer. Input must be a Pandas dataframe")
            exit(10)
        
    
    def fit(self, X, y=None):
        
        ##Check if input object is a pandas df, else raise exception
        self.check_input_obj(X,'fit()')

        ##Declare empty column_names 
        self.column_names = [] 
        
        ##Create an empty dict, which will hold the fitted LabelBinarizer for each column
        self.binarizers={}
        
        for col in X.columns:
            #Get the unique elements of the column
            uniq_elements = list(set(X[col]))

            #Determine the length of the unique elements.
            #If we have only 2 elements, then LabelBinarizer will create only one column
            #however if we have more than 2 unique elements then LabelBinarizer will create
            #a matrix with same number of columns as the number of unique elements.
            #So we will add a dummy element to make the distinct levels of the category variable as 3,
            #and this will make the LabelBinarizer to get 3 column matrix to represent the classes.
            #Later we will only consider the columns related to the two levels, and ignore the column related to the dummy element.            
            if len(uniq_elements) == 2:
               ##Add a dummy class
               #We have to name this class in a weird fashion,so that no data has this class
               uniq_elements.append('d#u/m*m-y+class_991-+xya')
            
            #Fit a LabelBinarizer on the unique elements
            lb = LabelBinarizer()
            self.binarizers[col] = lb.fit(uniq_elements)
            #Collect the column names. Discard the column name related to the dummy level (if any).
            #Also prefix column name to the actual classes in the column, so that we will have unique columns
            
            self.column_names = self.column_names + [str(col) + "_" + str(j) for j in list(lb.classes_) if j != 'd#u/m*m-y+class_991-+xya']
        return self
    
    def transform(self, X, y=None):
            self.check_input_obj(X,'transform()')
            temp_transformed_data = []
            transformed_column_names = []
            #For each key in the dictionary, use the associated model to transform the column in X
            for key, value in self.binarizers.items():
                #Handle the condition so that our transformations will continue, even though the 
                #input data frame does not have the key (or column)
                try:
                   temp_transformed_data.append(value.transform(X[key]))
                   transformed_column_names = transformed_column_names + [str(key) + "_" + str(j) for j in list(value.classes_)]
                except:
                   continue                
            #Prepare a Pandas data frame and select only the relevant columns, and discard the rest       
            return pd.DataFrame(np.concatenate(temp_transformed_data, axis=1),columns=transformed_column_names)[self.column_names]

            
##To test the above transformer... call the below function to test:
def test_CatMultiLabelTransformer(input=True):
  if input:
    ct = CatMultiLabelTransformer()
    demo_df_1=pd.DataFrame(['red','green'],columns=['Color'])
    ct.fit(demo_df_1)
    print("\n\nTraining data:\n{}".format(demo_df_1))
    print("\n\nData to be transformed is given below:.\nObserve that we have new class 'blue'")
    print("'blue' was not existing the training data.\n\n")
    demo_df_2 = pd.DataFrame(['red','green','blue'],columns=['Color'])
    print(demo_df_2)
    print("\nAfter transforming/encoding...\n")
    print(pd.DataFrame(ct.transform(demo_df_2),columns=ct.column_names))
    print("But if we have 3 classes in the training data:")
    print("Another training data set:")
    demo_df_1=pd.DataFrame(['S','M','XL'],columns=['Sizes'])
    ct.fit(demo_df_1)
    print("\n\nTraining data:\n{}".format(demo_df_1))
    print("Transformed data:")
    print(pd.DataFrame(ct.transform(demo_df_1),columns=ct.column_names))
  else:
    pass  

#To test CatMultiLabelTransformer, call test_CatMultiLabelTransformer(input=True)      
test_CatMultiLabelTransformer(input=False)


##The ScaleData will help us to evaluate which scale method performs better for numeric columns
##We will use std scaler by default, but another option is min-max scaler.
##The constructor parm std_scaler can be used to control this option while 
##testing the models using CV method.            
class ScaleData(BaseEstimator, TransformerMixin):
    def __init__(self,std_scaler=True):
        self.std_scaler = std_scaler

    def fit(self,X, y=None):

        if self.std_scaler == True:
           self.scaler = StandardScaler()
        else:
           self.scaler = MinMaxScaler()
        return self.scaler.fit(X)

    def transform(self,X):
        return self.scaler.transform(X)


##Do not include the following pipelines in a function, so that these can be available 
##if this code is imported as a package, and not executed as a program. See score_model.py 
##for more info.

        
#Numeric columns. We will ignore passenger ID and Ticket            
numeric_columns = ['Age','SibSp','Parch','Fare']        

#Categorical columns except the Cabin column        
categorical_columns = ['Pclass','Sex','Embarked']

#Define a pipeline for numeric variables transformation
#Age is the only column with Null values, but will still impute for all
#Numerical variables, since we do not know if the input numerical data contains 
#unavailable data after we deploy the model in Production


#Pipeline to handle numeric columns
#We are using the default impute strategy as 'median', but we will test 'average' and 'most frequently' also
numeric_pipeline = Pipeline([('selector',DataFrameSelector(numeric_columns)),
                            ('imputer',Imputer(strategy='median')),
                            ('scaledata',ScaleData(std_scaler=False))])

#Pipeline to handle Cabin column                             
cabin_pipeline = Pipeline([('selector',DataFrameSelector(['Cabin'])),
                           ('CabinTransformer',CabinTransformer(strategy='binomial'))
                          ])

#Pipeline to handle categorical columns (except cabin column)
categorical_pipeline = Pipeline([('selector',DataFrameSelector(categorical_columns)),
                                 ('fillnull',CatNullTransformer(value='Null')),
                                 ('CatMultiLabelTransformer',CatMultiLabelTransformer())
                                ])

#Combine all the transformations                                
full_transform_pipeline = FeatureUnion(transformer_list=[
           ("numeric_pipeline",numeric_pipeline),
           ("cabin_pipeline",cabin_pipeline),
           ("categorical_pipeline",categorical_pipeline)
           ])            

def main():
        ##In this block we will actually get the train data, perform transformations and
        ##fit the best model based on accuracy of 5 fold CV test.
        
        ##Read the file
        df = read_file('./train.csv')
        
        ##Get the training data
        X_train,y_train = get_train_data(df)
        
        ##Display sample rows of training data after the final transformation
        training_data_transformed = full_transform_pipeline.fit_transform(X_train)
        
        ##Get the column names:
        ##Observe how we are referencing the object variable using named_steps
        training_variables = numeric_pipeline.named_steps['selector'].column_names + cabin_pipeline.named_steps['CabinTransformer'].column_names + \
                             categorical_pipeline.named_steps['CatMultiLabelTransformer'].column_names
        print("\n\nTransformed data (initial 5 rows only), which is ready for training is displayed below:\n")
        print(pd.DataFrame(training_data_transformed,columns = training_variables).head(5))

        ##Pipeline for random forest model
        rf_prepare_and_train = Pipeline([('full_transform_pipeline',full_transform_pipeline),
                                   ('rf',RandomForestClassifier())
                                  ])

        ##Define a dictionary containing the list of hyper parms to be tested                          
        ##For random forest, we will test the CV score using 10,50,100 trees
        parm_grid = [{'full_transform_pipeline__cabin_pipeline__CabinTransformer__strategy': ['drop','binomial'], 
                      'full_transform_pipeline__numeric_pipeline__imputer__strategy':['mean','median','most_frequent'],
                      'full_transform_pipeline__numeric_pipeline__scaledata__std_scaler':[True,False],
                      'rf__n_estimators':[10,50,100]}]

        ##Perform grid search with a CV of 5
        ##Change verbose=2, to display detailed report
        grid_search = GridSearchCV(rf_prepare_and_train,parm_grid,cv=5,scoring='accuracy',verbose=1)
        grid_search.fit(X_train,y_train)              

        print("Best hyper parameters using Random Forests:")
        print(grid_search.best_params_)

        print("Best 5 fold CV score (accuracy):")
        print(grid_search.best_score_)

        final_model = grid_search.best_estimator_
        print(final_model)
        final_model.fit(X_train,y_train)
        print("Saving the best model..")
        joblib.dump(final_model,"rf_best_model.pkl")
        print("Saving Complete!")


##Boiler plate
if __name__ == '__main__':
    main()

                                