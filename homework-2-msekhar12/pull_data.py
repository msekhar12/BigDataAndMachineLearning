#################################
##Program Name: pull_data.py   ##
##Python version: 3.4          ##
##Author: Sekhar Mekala        ##
#################################

#Import the required packages

import os
import requests
import getpass
import platform

def os_level_config():
    ##The following statement is needed for Ubuntu env only
    if platform.system() == 'Linux':
        os.environ['REQUESTS_CA_BUNDLE']=os.path.join('/etc/ssl/certs/','ca-certificates.crt')
        
        ##The following 2 statements help to ignore security warnings.
        ##I was getting the warnings in Linux only. so disable only
        ##for Linux env
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        

def get_credentials(file = 'credentials.txt'):
    '''
       Input: File with userID and password details, both separated by a colon, as shown below:
       userid:password
       Output: [User ID, Password]
    '''
    #Open the credentials file and read the contents
    try:
      with open(file) as f:
         #Consider the lines which have a colon
         credentials = [x.strip().split(':') for x in f.readlines() if ':' in x]
         
         #Raise an exception if the credentials file is empty
         #And force user to enter credentials
         if len(credentials) == 0:
            raise ValueError
    except:
        #You did not supply the credentials.txt file, so prompt for username and password!!
        username=input("Enter your user name: ")
        password=getpass.getpass("Enter your password: ")
        return [username,password]
        
    
    #Only return the first parsed line.    
    return credentials[0]



def fetch_data(url,username=None, password=None,local_path='./',save_as='myfile.csv'):
    '''
    Input:
       url: URL 
       username: User ID
       password: Password
       local_path: Directory location to save the downloaded file
       save_as: File name to save the downloaded file
    '''
    
    #Check if the path exists in the local file system, if does not exist create the directory
    if not os.path.isdir(local_path):
        os.makedirs(local_path)
    
    #Join the directory and file name to get the complete file path    
    file_path = os.path.join(local_path,save_as)
    
    #Create a dictionary with login credentials
    login_info =  {'UserName': username, 'Password': password}
    
    #Initial request
    init_request = requests.get(url)
    
    #Check the return code. If 200 then OK.
    if init_request.status_code == 200:
        final_request = requests.post(init_request.url, data = login_info)
    else:
       print("Error occured while opening the connection to the URL")
       return exit(10)
    #Check the content type of the file.
    if final_request.headers['Content-Type'] == 'text/csv':
        #Save the file    
        with open(file_path, 'wb') as f:
           f.write(final_request.content)
           print("Successfully downloaded the file and saved as ",file_path)
    else:
       print("Could not download the file. Check your user name/password.")
       exit(10)       

def download(urls,local_path='./'):       
    '''
       urls is a dictionary in the following format:
       {'file_name1':'url-1','file_name2':'url-2'...}
       
       local_path is the directory where the files have toe be saved.
    '''
    
    #Initialoze username and password to empty strings
    username='' 
    password=''

    #Check if credentials.txt file is supplied,
    #else iterate till you get some values for username and password 
    while len(username) == 0 or len(password) == 0:        
          username, password = get_credentials(file = 'credentials.txt')
          if len(username) == 0 or len(password) == 0:
             choice = input("User name/Password is missing. To exit type EXIT. To continue press ENTER\n")
             if choice.lower() == 'exit':
                print("Good bye!!")
                exit(0)
             else:
               continue
               
    for key, value in urls.items():
         print("Downloading "+key+" data to the ",local_path," directory", "... ")
         fetch_data(value,username=username, password=password,local_path=local_path,save_as=key)    
          

def main():
    
    #Prepare the urls dictionary.
    #Dictionary's keys will be the file names to be used to save the downloaded data
    #Dictionary's values will be the file locations
    urls = {"train.csv":"https://www.kaggle.com/c/titanic/download/train.csv", \
            "test.csv": "https://www.kaggle.com/c/titanic/download/test.csv"}
    
    #Check the OS and perform some setup, if OS is Linux    
    os_level_config()
    
    #Start the download
    download(urls)

##Boiler plate. 
##Call if and only if this code is executed as a program.
if __name__ == '__main__':
    main()
    