import csv  
a = [ ]   
with open('covid.csv', 'r') as csvfile:     
    for row in csv.reader(csvfile):         
        a.append(row) 
        print(row)       
print ("\n The total number of training inwstances are : ", len(a)-1)   

num_attribute = len(a[0])-1  
print ("\n The initial hypothesis is : ")  
hypothesis = ['0'] * num_attribute 
print(hypothesis)   

for i in range(1, len(a)):     
    if a[i][num_attribute] == 'positive':         
        for j in range(0, num_attribute):              
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:                  
                hypothesis[j] = a[i][j]              
            else:                  
                hypothesis[j] = '?'      
    print ("\n The hypothesis for the training instance {} is : \n"  .format(i),   
                hypothesis)  
    
print("\n The final hypothesis  is ")  
print(hypothesis) 
