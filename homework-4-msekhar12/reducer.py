import sys

def reducer():
    #Initialize the variables
    product = None
    sale_total = 0
    
    for line in sys.stdin:
        data = line.strip().split("\t")
        #ignore the line if more than 2 fields
        if len(data) != 2:
            continue
        #Check that wer are not in the initial state    
        if product != None and product != data[0]:
           print "{0},{1}".format(product,sale_total)
           sale_total = 0
        product = data[0]
        sale_total = sale_total + float(data[1])
    #edge condition check and writing the final sum    
    if product != None:
       print "{0},{1}".format(product,sale_total)
def main():
    reducer()

if __name__ == '__main__':
    main()