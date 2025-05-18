import pandas as pd 

data = { 
'Fever': ['Y','Y', 'N','Y','N'], 
'Cough': ['Y','N','Y','Y','N'], 
'Throat Pain': ['Y','Y', 'N','Y','N'], 
'Body Pain':['Y','Y', 'N','N','Y'], 
'Covid-19':['positive','positive','negative','positive','negative'] 
}  


# data = {
#     "Sky": ["Sunny", "Sunny", "Rainy", "Sunny"],
#     "AirTemp":["Normal", "High", "High", "High"],
#     "Wind": ["Strong", "Strong", "Strong", "Strong"],
#     "Water": ["Warm", "Warm", "Warm", "Cold"],
#     "Forcast": ["Same", "Same", "Change", "Change"],
#     "EnjoySport": ["Yes", "Yes", "No", "Yes"]
# }

df = pd.DataFrame(data) 

df.to_csv('covid.csv', index=False) 
# df.to_csv('sport.csv', index=False) 
print("CSV file created successfully!")