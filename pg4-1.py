import pandas as pd 

data = { 
'Fever': ['Y','Y', 'N','Y','N'], 
'Cough': ['Y','N','Y','Y','N'], 
'Throat Pain': ['Y','Y', 'N','Y','N'], 
'Body Pain':['Y','Y', 'N','N','Y'], 
'Covid-19':['positive','positive','negative','positive','negative'] 
}  

df = pd.DataFrame(data) 

df.to_csv('covid.csv', index=False) 
print("CSV file created successfully!")
