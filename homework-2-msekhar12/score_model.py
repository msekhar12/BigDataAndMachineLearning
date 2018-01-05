#################################
##Program Name: score_model.py ##
##Python version: 3.4          ##
##Author: Sekhar Mekala        ##
#################################

##The following 2 statements are needed,
##to run the program on ubuntu env
import matplotlib
matplotlib.use('Agg')

##We are importing train_model.py as a "package".
##So the training episode will not happen in this program.
from train_model import *
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_conf_matrix(y_test,y_test_predict,model):

        ##Generate the confusion matrix
        cm = confusion_matrix(y_test, y_test_predict)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # view with a heatmap
        sns.heatmap(cm, annot=True, annot_kws={"size":30}, 
                    cmap='Blues', square=True, fmt='.3f')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion matrix for:\n{}'.format(model))
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix is saved as confusion_matrix.png!!")

def print_and_save_report(y_test,y_test_predict,model):
       print("\nScores of "+model+":")        
       print("Test f1 score:{}".format(f1_score(y_test,y_test_predict)))
       print("Test precision score:{}".format(precision_score(y_test,y_test_predict)))
       print("Test recall score:{}".format(recall_score(y_test,y_test_predict)))
       print("Test accuracy score:{}".format(accuracy_score(y_test,y_test_predict)))
       print("\n Saving the results to accuracy_report.txt...")
       with open('accuracy_report.txt','w') as f:
             f.write("Test f1 score:{}".format(f1_score(y_test,y_test_predict)))
             f.write("\nTest precision score:{}".format(precision_score(y_test,y_test_predict)))
             f.write("\nTest recall score:{}".format(recall_score(y_test,y_test_predict)))
             f.write("\nTest accuracy score:{}".format(accuracy_score(y_test,y_test_predict)))
       print("\nAccuracy report saved!!")     


def main():
        ##Read the saved model
        rf_model = joblib.load("rf_best_model.pkl")
        ##Read the file (using the function in the package train_model)
        X_test = read_file('./test_data.csv')
        y_test = X_test['Survived']
        X_test = X_test.drop(['Survived'],axis=1)

        y_test_predict = rf_model.predict(X_test)
        print_and_save_report(y_test,y_test_predict,'Random Forest')
        plot_conf_matrix(y_test,y_test_predict,'Random Forest')
        
##Boiler plate syntax        
if __name__ == '__main__':
    main()