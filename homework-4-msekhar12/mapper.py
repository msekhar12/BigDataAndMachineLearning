import sys

def mapper():
    for line in sys.stdin:
        data = line.strip().split("\t")
        #Make sure that number of fields are 6, else ignore
        if len(data) != 6:
            continue
        #Extract the desired fields    
        category = data[3]
        price = data[4]
        #write to std o/p as a tab delimited data
        print "{0}\t{1}".format(category,price)

def main():
    mapper()

#Boiler plate syntax    
if __name__ == '__main__':
   main()