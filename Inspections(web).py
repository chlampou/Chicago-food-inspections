# In order to view the interactive folium plots, 
# if you use Spyder (Anaconda)
# first you need to left click to Files,
# then scroll down until the end
# and then right click to the file you want to open
# and select Open externally  


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import folium
import webbrowser
from folium import plugins
from folium.plugins import FastMarkerCluster
import plotly
import plotly.graph_objs as go

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import cv

from wordcloud import WordCloud
from collections import Counter
from PIL import Image

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

import shap


 
# Import the food inspections dataset (~3mins to run)
df_food = pd.read_csv('https://data.cityofchicago.org/resource/4ijn-s7e5.csv?$limit=300000', parse_dates=['inspection_date'])
print('The df_food before data cleaning')
print(df_food.head())

# Check dataset
print(df_food.dtypes.sort_values())
print(df_food.shape)

# Rename the specific column
df_food.rename(columns={'license_': 'license'}, inplace=True)

# Check if records are related to Illinois
print(df_food['state'].unique())
# Keep only the records for Illinois
df_food=df_food[df_food['state']=='IL']

# Drop the specific columns 
df_food = df_food.drop('state', 1)
df_food = df_food.drop('aka_name', 1)
df_food = df_food.drop('location', 1)

# Check for duplicates based on inspection_id
print('Df_food has', df_food.duplicated(subset=['inspection_id']).sum(), 'duplicate values based on inspection_id')

# Check for missing values
print('Missing values')
print(df_food.isnull().sum().sort_values(ascending=False))

# Drop rows which miss dba_name or risk or license
# or inspection_type or facility_type 
# or longitude or latitude
df_food=df_food.dropna(subset=['dba_name'])
df_food=df_food.dropna(subset=['risk'])
df_food=df_food.dropna(subset=['license'])
df_food=df_food.dropna(subset=['inspection_type'])
df_food=df_food.dropna(subset=['facility_type'])
df_food=df_food.dropna(subset=['longitude'])
df_food=df_food.dropna(subset=['latitude'])

# Check the column facility_type
print(df_food['facility_type'].unique())

# Top 10 risky facility types
c1=sns.barplot(x=df_food['facility_type'].value_counts()[:10],y=df_food['facility_type'].value_counts()[:10].index)
c1.set_title('Top 10 risky facility types')
c1.set_ylabel('')
c1.set_xlabel('')
plt.show()

# Keep only records about Restaurants and Grocery stores
df_food = df_food[df_food.facility_type.isin(['Restaurant', 'Grocery Store'])]

# Check the column inspection_type
print(df_food['inspection_type'].unique())

# Top 10 inspection types
c2=sns.barplot(x=df_food['inspection_type'].value_counts()[:10],y=df_food['inspection_type'].value_counts()[:10].index)
c2.set_title('Top 10 inspection types')
c2.set_ylabel('')
c2.set_xlabel('')
plt.show()

# Keep only records about Canvass, Complaint and their Re-inspections 
df_food = df_food[df_food.inspection_type.isin(['Canvass', 'Complaint', 'Canvass Re-Inspection', 'Complaint Re-Inspection'])]

# Check the column results
print(df_food['results'].unique())
# Keep only successfull inspections
df_food = df_food[~df_food.results.isin(['Out of Business', 'Business Not Located', 'No Entry', 'Not Ready'])]

# Inspection results
value_counts = df_food['results'].value_counts()
value_counts.plot.bar(title = 'Inspection results', color=['green', 'green', 'red'])
plt.xticks(rotation=0)
plt.show()

# Replace the values in order to have only Pass and Fail
df_food['results']=df_food['results'].replace(['Pass w/ Conditions'],['Pass'])

# Check the column risk
print(df_food['risk'].unique())
# Drop rows with value All
df_all = df_food[df_food['risk']=='All']
df_food=df_food.drop(df_all.index, axis=0)

# Risk results
value_counts = df_food['risk'].value_counts()
value_counts.plot.bar(title = 'Risk', color=['red', 'orange', 'green'])
plt.xticks(rotation=0)
plt.show()

# Create new variables: year, month and day (from extracting the inspection_date)
df_food['year'] = pd.DatetimeIndex(df_food['inspection_date']).year
df_food['month'] = pd.DatetimeIndex(df_food['inspection_date']).month
df_food['day'] = pd.DatetimeIndex(df_food['inspection_date']).day

# Drop missing values for the specific column
df_food=df_food.dropna(subset=['zip'])

print('The df_food after data cleaning')
print(df_food.dtypes.sort_values())
print(df_food.shape)
print('Fixed missing values')
print(df_food.isnull().sum().sort_values(ascending=False))
print("------------------------------------------------------------")





# NLP for Violations description (~5mins to run)
print(df_food.iloc[10000].violations)

# Split violations into binary values for each violation
def split_violations(violations):
    values_row = pd.Series([])
    if type(violations) == str:
        violations = violations.split(' | ')
        for violation in violations:
            index = "v_" + violation.split('.')[0]
            values_row[index] = 1
    return values_row

# Calculate violation values, set missing violations to 0
values_data = df_food.violations.apply(split_violations).fillna(0)

# Generate column names
critical_columns = [("v_" + str(num)) for num in range(1, 30)]
serious_columns = [("v_" + str(num)) for num in range(30, 50)]
minor_columns = [("v_" + str(num)) for num in range(50, 65)]
minor_columns.append("v_70")

# Create complete list of column names
columns = critical_columns + serious_columns + minor_columns

# Create the dataframe 'values' by using column names, violation data and inspection_id
values = pd.DataFrame(values_data, columns=columns)
values['inspection_id'] = df_food['inspection_id']

# Display 'values' dataframe
print('Values has', values.shape)
print(values.head())



# Create the dataframe 'counts' by counting the number of violations of each category for each inspection
counts = pd.DataFrame({
    'critical_count': values[critical_columns].sum(axis=1),
    'serious_count': values[serious_columns].sum(axis=1),
    'minor_count': values[minor_columns].sum(axis=1)
})

counts['violation_count']=counts['critical_count']+counts['serious_count']+counts['minor_count'] 

# Check if critical violation found
counts.loc[counts['critical_count'] == 0, 'critical_found'] = 0 
counts.loc[counts['critical_count'] != 0, 'critical_found'] = 1 

counts['inspection_id'] = df_food['inspection_id']

counts['crit_enc']=counts['critical_found'].replace([1,0],['Yes', 'No'])

# Critical found
value_counts = counts['crit_enc'].value_counts()
value_counts.plot.bar(title = 'Critical found', color=['green', 'red'])
plt.xticks(rotation=0)
plt.show()

counts = counts.drop('crit_enc', 1) 

# Display 'counts' dataframe
print('Counts has', counts.shape)
print(counts.head())





# Plots
# The distribution of the variable zip
a=df_food['zip'].plot(kind='hist')
a.set_title('The distribution of the zip codes')
a.set_ylabel('')
a.set_xlabel('')
plt.show()

# Top 10 businesses with their distribution of inspections
b=sns.barplot(x=df_food['dba_name'].value_counts()[:10],y=df_food['dba_name'].value_counts()[:10].index)
b.set_title('Top 10 businesses')
b.set_ylabel('')
b.set_xlabel('')
plt.show()

# Risk - Results
plt.figure(figsize=(10,10))
plt.title('Risk - Results')
c=sns.heatmap(pd.crosstab([df_food.risk], [df_food.results]), square=True,
            cmap='Spectral', annot=True, fmt='.1f', linewidths=0.5, cbar=False)
c.set_ylabel('')
c.set_xlabel('')
plt.show()

# Inspections per year
x=df_food.year.value_counts().index
y=df_food.year.value_counts()
d1=sns.barplot(x=x,y=y, color='b', alpha=0.7)
d1.set_title('Inspection counts by year')
d1.set_ylabel('')
d1.set_xlabel('')
plt.show()

# Inspections per month
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

x=df_food.month.value_counts().index
y=df_food.month.value_counts()
fig, ax = plt.subplots()
d2=sns.barplot(x=x,y=y,color='b', alpha=0.7)
d2.set_title('Inspection counts by month')
ax.set_xticklabels(months)
plt.xticks(rotation=45)
d2.set_ylabel('')
plt.show()

# Risk per year
e1=df_food.groupby('year').risk.value_counts().unstack().plot.barh()
e1.set_title('Risk by year')
e1.set_ylabel('')
e1.set_xlabel('')
plt.show()

# Risk per month
e2=df_food.groupby('month').risk.value_counts().unstack().plot.barh()
e2.set_title('Risk by month')
e2.set_ylabel('')
e2.set_xlabel('')
plt.show()

# Results per year
f1=df_food.groupby('year').results.value_counts().unstack().plot.barh()
f1.set_title('Results by year')
f1.set_ylabel('')
f1.set_xlabel('')
plt.show()

# Results per month
f2=df_food.groupby('month').results.value_counts().unstack().plot.barh()
f2.set_title('Results by month')
f2.set_ylabel('')
f2.set_xlabel('')
plt.show()



lats = df_food['latitude'].tolist()
lons = df_food['longitude'].tolist()

print('The minimum value for the latitude is: '+str(min(lats)))
print('The maximum value for the latitude is: '+str(max(lats)))
print('The minimum value for the longitude is: '+str(min(lons)))
print('The maximum value for the longitude is: '+str(max(lons)))

locations = list(zip(lats, lons))
map1 = folium.Map(location=[37.0902, -95.7129], zoom_start=5)
FastMarkerCluster(data=locations).add_to(map1)
map1.save('map1.html')


map2 = folium.Map([41.8600, -87.6298], zoom_start=10)
# Convert to (n, 2) nd-array format for heatmap
inspections_arr = df_food.sample(20000)[['latitude', 'longitude']].values
# Plot heatmap
map2.add_child(plugins.HeatMap(inspections_arr.tolist(), radius=10))
map2.save('map2.html')



# Health code violations
titles = pd.DataFrame({
    "v_1": "Person in charge present, demonstrates knowledge, and performs duties (1)",
    "v_2": "City of Chicago Food Service Sanitation Certificate (2)",
    "v_3": "Management, food employee and conditional employee; knowledge, responsibilities and reporting (3)",
    "v_4": "Proper use of restriction and exclusion (4)",
    "v_5": "Procedures for responding to vomiting and diarrheal events (5)",
    "v_6": "Proper eating, tasting, drinking, or tobacco use (6)",
    "v_7": "No discharge from eyes, nose, and mouth (7)",
    "v_8": "Hands clean & properly washed (8)",
    "v_9": "No bare hand contact with RTE food or a pre-approved alternative procedure properly allowed (9)",
    "v_10": "Adequate handwashing sinks properly supplied and accessible (10)",
    "v_11": "Food obtained from approved source (11)",
    "v_12": "Food received at proper temperature (12)",
    "v_13": "Food in good condition, safe, & unadulterated (13)",
    "v_14": "Required records available: shellstock tags, parasite (14)",
    "v_15": "Food separated and protected (15)",
    "v_16": "Food-contact surfaces: cleaned & sanitized  (16)",
    "v_17": "Proper disposition of returned, previously served, reconditioned & unsafe food (17)",
    "v_18": "Proper cooking time & temperatures  (18)",
    "v_19": "Proper reheating procedures for hot holding (19)",
    "v_20": "Proper cooling time and temperature (20)",
    "v_21": "Proper hot holding temperatures (21)",
    "v_22": "Proper cold holding temperatures (22)",
    "v_23": "Proper date marking and disposition (23)",
    "v_24": "Time as a Public Health Control; procedures & records (24)",
    "v_25": "Consumer advisory provided for raw/undercooked food (25)",
    "v_26": "Pasteurized foods used; prohibited foods not offered (26)",
    "v_27": "Food additives: approved and properly used (27)",
    "v_28": "Toxic substances properly identified, stored, & used (28)",
    "v_29": "Compliance with variance/specialized process/HACCP (29)",
    "v_30": "Pasteurized eggs used where required (30)",
    "v_31": "Water & ice from approved source (31)",
    "v_32": "Variance obtained for specialized processing methods (32)",
    "v_33": "Proper cooling methods used; adequate equipment for temperature control (33)",
    "v_34": "Plant food properly cooked for hot holding (34)",
    "v_35": "Approved thawing methods used (35)",
    "v_36": "Thermometers provided & accurate (36)",
    "v_37": "Food properly labeled; original containe (37)",
    "v_38": "Insects, rodents, & animals not present (38)",
    "v_39": "Contamination prevented during food preparation, storage & display (39)",
    "v_40": "Personal cleanliness (40)",
    "v_41": "Wiping cloths: properly used & stored (41)",
    "v_42": "Washing fruits & vegetables (42)",
    "v_43": "In-use utensils: properly stored (43)",
    "v_44": "Utensils, equipment & linens: properly stored, dried, & handled (44)",  
    "v_45": "Single-use/single-service articles: properly stored & used (45)",
    "v_46": "Gloves used properly (46)",
    "v_47": "Food & non-food contact surfaces cleanable, properly designed, constructed & used (47)",
    "v_48": "Warewashing facilities: installed, maintained & used; test strips (48)",
    "v_49": "Non-food contact surfaces clean (49)",
    "v_50": "Hot & cold water available; adequate pressure (50)",
    "v_51": "Plumbing installed; proper backflow devices (51)",
    "v_52": "Sewage & waste water properly disposed (52)",
    "v_53": "Toilet facilities: properly constructed, supplied, & cleaned (53)",
    "v_54": "Garbage & refuse properly disposed; facilities maintained (54)",
    "v_55": "Physical facilities installed, maintained & clean (55)",
    "v_56": "Adequate ventilation & lighting; designated areas used (56)",
    "v_57": "IN OUT     All food employees have food handler training (57)",
    "v_58": "IN OUT     Allergen training as required (58)",
    "v_59": "Previous priority foundation violation corrected (59)",
    "v_60": "Previous core violation corrected (60)",
    "v_61": "Summary Report displayed and visible to the public (61)",
    "v_62": "Compliance with Clean Indoor Air Ordinance (62)",
    "v_63": "Removal of Suspension Sign (63)",    
    "v_64": "Public Health nuisance (64)",   
    "v_70": "No smoking regulations (70)"
}, index=[0])

# Change the name of columns in value dataframe by the title values dataframe's columns
titled_values = values.rename(columns=titles.iloc[0])

# Sum binary values for each violation
sums = titled_values.drop('inspection_id', axis=1).sum()

# Generate color list
colors = ['red']*29 + ['orange']*20 + ['green']*16

# Sort sums and colors by sum value
sum_data = pd.DataFrame({'sums': sums, 'colors': colors}).sort_values('sums')

plt.rcParams['figure.figsize'] = (15, 15)
ax = sum_data.sums.plot(kind='barh', color=sum_data.colors)
ax.set_title('Health Code Violations')
ax.invert_yaxis()
plt.show()

# Top 10 health code violations
top10_data=sum_data.sort_values('sums', ascending=False)
colors = top10_data.colors
plt.rcParams['figure.figsize'] = (10, 10)
ax = top10_data.sums[:10].plot(kind='barh', color=colors)
ax.set_title('Top 10 violations')
ax.invert_yaxis()
plt.show()

# In order to display the next image, first you need to downnload  
# and save the image from this link: https://prnt.sc/1uc23mb
# the result must be something like this: https://prnt.sc/1uhg47d
mask = np.array(Image.open('C:/Users/User/Desktop/Diploma/fast_food.png'))

# Extract comments from violations (~1min to run)
def get_comments(violations):
    comments = ""
    if type(violations) == str:
        violations = violations.split(' | ')
        for violation in violations:
            violation = violation.split('Comments:')
            if len(violation) == 2:
                comments += violation[1]
    return comments

# Concatenate all comments
comments = df_food.violations.apply(get_comments).str.cat(sep=" ")

# Generate wordcloud
comments_wordcloud = WordCloud(background_color='white', mask=mask, colormap='Dark2',
                               mode='RGB', width=2000, max_words=1000, height=2000, contour_width=1, 
                               contour_color='steelblue').generate(comments)

# Plot wordcloud
plt.rcParams['figure.figsize'] = (10, 10)
plt.imshow(comments_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()










# Import the business licenses dataset (~5mins to run)
df_business = pd.read_csv('https://data.cityofchicago.org/resource/r5kz-chrr.csv?$limit=1500000', parse_dates=['license_start_date', 'expiration_date'])
print('The df_business before data cleaning')
print(df_business.head())

# Check dataset
print(df_business.dtypes.sort_values())
print(df_business.shape)

# Check for duplicates 
print('Business has', df_business.duplicated(subset=['id']).sum(), 'duplicated values')
df_business=df_business.drop_duplicates(subset=['id'])
print('Business has', df_business.duplicated(subset=['id']).sum(), 'duplicated values')

# Check for missing values
print('Missing values')
print(df_business.isnull().sum().sort_values(ascending=False))

# Keep only specific columns
df_business=df_business[['license_start_date','state', 'doing_business_as_name', 'address', 'id', 
                         'license_number', 'license_description', 'expiration_date']]

# Rename the columns 
df_business.rename(columns={'expiration_date': 'license_expiration_date'}, inplace=True)
df_business.rename(columns={'license_number': 'license'}, inplace=True)
df_business.rename(columns={'doing_business_as_name': 'dba_name'}, inplace=True)

# Check if records are related to Illinois
print(df_business['state'].unique())
# Keep only the records for Illinois
df_business=df_business[df_business['state']=='IL']
df_business = df_business.drop('state', 1) 

# Drop rows which miss license or dba_name
# or license_start_date or license_expiration_date
df_business=df_business.dropna(subset=['license'])
df_business=df_business.dropna(subset=['dba_name'])
df_business=df_business.dropna(subset=['license_start_date'])
df_business=df_business.dropna(subset=['license_expiration_date'])

# Check license_description
print(df_business['license_description'].unique())
a1=sns.barplot(x=df_business['license_description'].value_counts()[:10],y=df_business['license_description'].value_counts()[:10].index)
a1.set_title('Top 10 license description')
a1.set_ylabel('')
a1.set_xlabel('')
plt.show()

print('The df_business after data cleaning')
print(df_business.dtypes.sort_values())
print(df_business.shape)
print('Fixed missing values')
print(df_business.isnull().sum().sort_values(ascending=False))
print("------------------------------------------------------------")










# Import the garbage complaints dataset (~2mins to run)
df_garbage = pd.read_csv('https://data.cityofchicago.org/resource/9ksk-na4q.csv?$limit=500000', parse_dates=['creation_date'])
print('The df_garbage before data cleaning')
print(df_garbage.head())

# Check dataset
print(df_garbage.dtypes.sort_values())
print(df_garbage.shape)

# Check for duplicates 
print('Garbage has', df_garbage.duplicated(subset=['service_request_number']).sum(), 'duplicated values')
df_garbage=df_garbage.drop_duplicates(subset=['service_request_number'])
print('Garbage has', df_garbage.duplicated(subset=['service_request_number']).sum(), 'duplicated values')

# Check for missing values
print('Missing values')
print(df_garbage.isnull().sum().sort_values(ascending=False))

# Keep only specific columns
df_garbage=df_garbage[['creation_date', 'zip_code', 'status']]

# Rename the columns 
df_garbage.rename(columns={'zip_code': 'zip'}, inplace=True)

# Check the column status
print(df_garbage['status'].unique())
# Keep only the records Completed and Open
df_garbage = df_garbage.drop(df_garbage[(df_garbage['status'] == 'Completed - Dup') | (df_garbage['status'] == 'Open - Dup')].index)
df_garbage = df_garbage.drop('status', 1) 

# Drop missing values for the specific column
df_garbage=df_garbage.dropna(subset=['zip'])

# Creat new variables: year and month (from extracting the creation_date)
df_garbage['year'] = pd.DatetimeIndex(df_garbage['creation_date']).year
df_garbage['month'] = pd.DatetimeIndex(df_garbage['creation_date']).month

# Keep only one record for each zip for each year for each month
df_garbage=df_garbage.groupby(['zip', 'year', 'month']).size().reset_index(name='count_garbage')
print(df_garbage.head(20))

# Calculate the garbage frequency per month
df_garbage['garbage_frequency'] = (df_garbage.count_garbage / 30).round(2)
print(df_garbage.head(20))
df_garbage = df_garbage.drop('count_garbage', 1) 


print('The df_garbage after data cleaning')
print(df_garbage.dtypes.sort_values())
print(df_garbage.shape)
print('Fixed missing values')
print(df_garbage.isnull().sum().sort_values(ascending=False))
print("------------------------------------------------------------")










# Import the sanitation complaints dataset (~2mins to run)
df_sanitation = pd.read_csv('https://data.cityofchicago.org/resource/me59-5fac.csv?$limit=250000', parse_dates=['creation_date'])
print('The df_sanitation before data cleaning')
print(df_sanitation.head())

# Check dataset
print(df_sanitation.dtypes.sort_values())
print(df_sanitation.shape)

# Check for duplicates 
print('Sanitation has', df_sanitation.duplicated(subset=['service_request_number']).sum(), 'duplicated values')
df_sanitation=df_sanitation.drop_duplicates(subset=['service_request_number'])
print('Sanitation has', df_sanitation.duplicated(subset=['service_request_number']).sum(), 'duplicated values')

# Check for missing values
print('Missing values')
print(df_sanitation.isnull().sum().sort_values(ascending=False))

# Keep only specific columns
df_sanitation=df_sanitation[['creation_date', 'zip_code', 'status']]

# Rename the columns 
df_sanitation.rename(columns={'zip_code': 'zip'}, inplace=True)

# Check the column status
print(df_sanitation['status'].unique())
# Keep only the records Completed and Open
df_sanitation = df_sanitation.drop(df_sanitation[(df_sanitation['status'] == 'Completed - Dup') | (df_sanitation['status'] == 'Open - Dup')].index)
df_sanitation = df_sanitation.drop('status', 1) 

# Drop missing values for the specific column
df_sanitation=df_sanitation.dropna(subset=['zip'])

# Creat new variables: year and month (from extracting the creation_date)
df_sanitation['year'] = pd.DatetimeIndex(df_sanitation['creation_date']).year
df_sanitation['month'] = pd.DatetimeIndex(df_sanitation['creation_date']).month

# Keep only one record for each zip for each year for each month
df_sanitation=df_sanitation.groupby(['zip', 'year', 'month']).size().reset_index(name='count_sanitation')
print(df_sanitation.head(20))

# Calculate the sanitation frequency per month
df_sanitation['sanitation_frequency'] = (df_sanitation.count_sanitation / 30).round(2)
print(df_sanitation.head(20))
df_sanitation = df_sanitation.drop('count_sanitation', 1) 

print('The df_sanitation after data cleaning')
print(df_sanitation.dtypes.sort_values())
print(df_sanitation.shape)
print('Fixed missing values')
print(df_sanitation.isnull().sum().sort_values(ascending=False))
print("------------------------------------------------------------")










# Import the rodent complaints dataset (~2mins to run)
df_rodent = pd.read_csv('https://data.cityofchicago.org/resource/97t6-zrhs.csv?$limit=450000', parse_dates=['creation_date'])
print('The df_rodent before data cleaning')
print(df_rodent.head())

# Check dataset
print(df_rodent.dtypes.sort_values())
print(df_rodent.shape)

# Check for duplicates 
print('Rodent has', df_rodent.duplicated(subset=['service_request_number']).sum(), 'duplicated values')
df_rodent=df_rodent.drop_duplicates(subset=['service_request_number'])
print('Rodent has', df_rodent.duplicated(subset=['service_request_number']).sum(), 'duplicated values')

# Check for missing values
print('Missing values')
print(df_rodent.isnull().sum().sort_values(ascending=False))

# Keep only specific columns
df_rodent=df_rodent[['creation_date', 'zip_code', 'status']]

# Rename the columns 
df_rodent.rename(columns={'zip_code': 'zip'}, inplace=True)

# Check the column status
print(df_rodent['status'].unique())
# Keep only the records Completed and Open
df_rodent = df_rodent.drop(df_rodent[(df_rodent['status'] == 'Completed - Dup') | (df_rodent['status'] == 'Open - Dup')].index)
df_rodent = df_rodent.drop('status', 1) 

# Drop missing values for the specific column
df_rodent=df_rodent.dropna(subset=['zip'])

# Creat new variables: year and month (from extracting the creation_date)
df_rodent['year'] = pd.DatetimeIndex(df_rodent['creation_date']).year
df_rodent['month'] = pd.DatetimeIndex(df_rodent['creation_date']).month

# Keep only one record for each zip for each year for each month
df_rodent=df_rodent.groupby(['zip', 'year', 'month']).size().reset_index(name='count_rodent')
print(df_rodent.head(20))

# Calculate the rodent frequency per month
df_rodent['rodent_frequency'] = (df_rodent.count_rodent / 30).round(2)
print(df_rodent.head(20))
df_rodent = df_rodent.drop('count_rodent', 1) 

print('The df_rodent after data cleaning')
print(df_rodent.dtypes.sort_values())
print(df_rodent.shape)
print('Fixed missing values')
print(df_rodent.isnull().sum().sort_values(ascending=False))
print("------------------------------------------------------------")










# Import the 311 service requests dataset (~30mins to run)
df_service = pd.read_csv('https://data.cityofchicago.org/resource/v6vf-nfxy.csv?$limit=6000000', parse_dates=['created_date'])
print('The df_service before data cleaning')
print(df_service.head())

# Check dataset
print(df_service.dtypes.sort_values())
print(df_service.shape)

# Check for duplicates 
print('Service has', df_service.duplicated(subset=['sr_number']).sum(), 'duplicated values')

# Check for missing values
print('Missing values')
print(df_service.isnull().sum().sort_values(ascending=False))

# Check if records are related to Illinois
print(df_service['state'].unique())
# Keep only the records for Illinois
df_service=df_service.dropna(subset=['state'])
# Drop the column state
df_service = df_service.drop('state', 1)

# Check the column city
print(df_service['city'].unique())
# Drop the column city
df_service = df_service.drop('city', 1) 

# Keep only specific columns
df_service=df_service[['created_date', 'zip_code', 'sr_type']]

# Rename the columns 
df_service.rename(columns={'created_date': 'creation_date'}, inplace=True)
df_service.rename(columns={'zip_code': 'zip'}, inplace=True)

# Drop missing values for the specific column 
df_service=df_service.dropna(subset=['zip'])

# Check the column sr_type
print(df_service['sr_type'].unique())

# Keep only the date part 
df_service['creation_date']=df_service['creation_date'].dt.date

# Creat new variables: year and month (from extracting the creation_date)
df_service['year'] = pd.DatetimeIndex(df_service['creation_date']).year
df_service['month'] = pd.DatetimeIndex(df_service['creation_date']).month

# Keep only records about specific sr_type
df_service = df_service[df_service.sr_type.isin(['Sanitation Code Violation', 'Garbage Cart Maintenance', 'Rodent Baiting/Rat Complaint'])]

# Create new column sanitate
df_service.loc[df_service['sr_type'] == 'Sanitation Code Violation', 'sanitate'] = 1 
df_service.loc[df_service['sr_type'] != 'Sanitation Code Violation', 'sanitate'] = 0 

# Create new column garbage
df_service.loc[df_service['sr_type'] == 'Garbage Cart Maintenance', 'garbage'] = 1 
df_service.loc[df_service['sr_type'] != 'Garbage Cart Maintenance', 'garbage'] = 0 

# Create new column rodent 
df_service.loc[df_service['sr_type'] == 'Rodent Baiting/Rat Complaint', 'rodent'] = 1 
df_service.loc[df_service['sr_type'] != 'Rodent Baiting/Rat Complaint', 'rodent'] = 0 

# Keep only one record for each zip for each year for each month
df_service=df_service.groupby(['zip', 'year', 'month'], as_index=False).sum()
print(df_service.head(20))

# Calculate the 311 calls frequency per month
df_service['garbage_percent'] = (df_service.garbage / 30).round(2)
df_service['sanitate_percent'] = (df_service.sanitate / 30).round(2)
df_service['rodent_percent'] = (df_service.rodent / 30).round(2)

df_service = df_service.drop('garbage', 1) 
df_service = df_service.drop('sanitate', 1) 
df_service = df_service.drop('rodent', 1) 

df_service['zip'] = df_service.zip.astype(float)

print('The df_service after data cleaning')
print(df_service.dtypes.sort_values())
print(df_service.shape)
print('Fixed missing values')
print(df_service.isnull().sum().sort_values(ascending=False))
print("------------------------------------------------------------")







# Check the type of columns
print(df_food.dtypes.sort_values())
lbl = preprocessing.LabelEncoder()

# Transform the type of column results
df_food['enc_results'] = lbl.fit_transform(df_food['results'].astype(str))
# Check if replace is needed
print('Results values') 
print(df_food['results'].head(15))
print('Encoded result values') 
print(df_food['enc_results'].head(15))
# No need for replace
df_food = df_food.drop('results', 1)
df_food.rename(columns={'enc_results': 'results'}, inplace=True)


# Transform the type of column risk 
df_food['enc_risk'] = lbl.fit_transform(df_food['risk'].astype(str))
# Check if replace is needed
print('Risk values')
print(df_food['risk'].head(15))
print('Encoded risk values') 
print(df_food['enc_risk'].head(15))
# Replace the values
df_food['enc_risk']=df_food['enc_risk'].replace({0:1, 1:2, 2:3})
df_food = df_food.drop('risk', 1)
df_food.rename(columns={'enc_risk': 'risk'}, inplace=True)







data = df_food.loc[:, ['inspection_id', 'license', 'inspection_date', 'results', 'risk',
                       'zip', 'year', 'month', 'latitude', 'longitude', 'dba_name', 'address']]
print(data.shape)


# Merge with 'values' and 'counts' dataframes
data = pd.merge(data, values, on='inspection_id')
data = pd.merge(data, counts, on='inspection_id')
print(data.shape)

# Merge with df_garbage
data=pd.merge(data, df_garbage, how='left', on=['zip', 'year', 'month'])
print(data.dtypes.sort_values())
print(data.shape)
print(data.columns.values)
data['garbage_frequency'] = data['garbage_frequency'].fillna(0)

# Merge with df_sanitation
data=pd.merge(data, df_sanitation, how='left', on=['zip', 'year', 'month'])
print(data.dtypes.sort_values())
print(data.shape)
print(data.columns.values)
data['sanitation_frequency'] = data['sanitation_frequency'].fillna(0)

# Merge with df_rodent
data=pd.merge(data, df_rodent, how='left', on=['zip', 'year', 'month'])
print(data.dtypes.sort_values())
print(data.shape)
print(data.columns.values)
data['rodent_frequency'] = data['rodent_frequency'].fillna(0)

# Merge with df_service
data=pd.merge(data, df_service, how='left', on=['zip', 'year', 'month'])
print(data.dtypes.sort_values())
print(data.shape)
print(data.columns.values)
print(data.isnull().sum().sort_values(ascending=False))

data.loc[data['sanitate_percent'].notnull(), 'sanitation_frequency'] = data['sanitate_percent']
data.loc[data['garbage_percent'].notnull(), 'garbage_frequency'] = data['garbage_percent']
data.loc[data['rodent_percent'].notnull(), 'rodent_frequency'] = data['rodent_percent']

data = data.drop('sanitate_percent', 1)
data = data.drop('garbage_percent', 1)
data = data.drop('rodent_percent', 1)

# Check for duplicates based on inspection_id
print('Data has', data.duplicated(subset=['inspection_id']).sum(), 'duplicated values')
data=data.drop_duplicates(subset=['inspection_id'])
print('Data has', data.duplicated(subset=['inspection_id']).sum(), 'duplicated values')

print('Data has', data.shape)
print('Missing values')
print(data.isnull().sum().sort_values(ascending=False))  
print(data.columns.values)
print("------------------------------------------------------------")



# Create new variable: inspection_order_back
data['inspection_order_back']=data.sort_values('inspection_date', ascending=False).groupby('license').cumcount()+1

# How many different licenses exist
print('There are', len(data['license'].unique()), 'unique licenses')

# Create new variable: counting 
counting = data['license'].value_counts()
print(counting.unique())
print(counting.head())

# Drop 0 licenses
data = data[~data.license.isin([0.0])]
print(data.shape)

# How many licenses have one inspection
x = 1
d = Counter(counting)
print('There are {} licenses with {} inspection'.format(d[x], x))

# Select records with more than one inspection 
data=data[data['license'].isin(counting.index[counting > 1])]
print('There are', len(data), 'cases with more than one inspection')

# Sort inspections by inspection_date and group by license 
license_groups = data.sort_values('inspection_date').groupby('license')

# Find previous inspections by shifting each sorted group
past_data = license_groups.shift(1)

# Add past info
data['previous_critical_count'] = past_data.critical_count
data['previous_serious_count'] = past_data.serious_count
data['previous_minor_count'] = past_data.minor_count
data['previous_violation_count'] = past_data.violation_count

data['previous_garbage_frequency']=past_data.garbage_frequency
data['previous_sanitation_frequency']=past_data.sanitation_frequency
data['previous_rodent_frequency']=past_data.rodent_frequency

data['previous_month']=past_data.month
data['previous_results'] = past_data.results

data['previous_critical_ratio']= (data['previous_critical_count']/data['previous_violation_count']).round(2)
data['previous_serious_ratio']= (data['previous_serious_count']/data['previous_violation_count']).round(2)
data['previous_minor_ratio']= (data['previous_minor_count']/data['previous_violation_count']).round(2)

# Check if critical violation found in previous inspection
data.loc[data['previous_critical_count'] == 0, 'previous_critical_found'] = 0 
data.loc[data['previous_critical_count'] > 0, 'previous_critical_found'] = 1 

# Check if serious violation found in previous inspection
data.loc[data['previous_serious_count'] == 0, 'previous_serious_found'] = 0 
data.loc[data['previous_serious_count'] > 0, 'previous_serious_found'] = 1 

# Select past violation values, remove past inspection_id
past_values = past_data[values.columns].drop('inspection_id', axis=1).add_prefix("p")

# Add past values to model data
data = data.join(past_values)

print(data.shape)
print(data.columns.values)



# Critical_found - Previous_critical_found
plt.figure(figsize=(10,10))
plt.title('Critical found - Previous critical found')
c=sns.heatmap(pd.crosstab([data.previous_critical_found], [data.critical_found]), square=True,
            cmap='Spectral', annot=True, fmt='.1f', linewidths=0.5, cbar=False)
plt.show()

# Results - Previous_results
plt.figure(figsize=(10,10))
plt.title('Results - Previous results')
c=sns.heatmap(pd.crosstab([data.previous_results], [data.results]), square=True,
            cmap='Spectral', annot=True, fmt='.1f', linewidths=0.5, cbar=False)
plt.show()

# Critical_found - Previous_results
plt.figure(figsize=(10,10))
plt.title('Critical found - Previous results')
c=sns.heatmap(pd.crosstab([data.previous_results], [data.critical_found]), square=True,
            cmap='Spectral', annot=True, fmt='.1f', linewidths=0.5, cbar=False)
plt.show()


# Calculate cross tabulation of results and previous results
chart = pd.crosstab(data.previous_results, data.results)
# Make Numpy array of total counts of previous(prior) fails and passes with the following(post) results 
chart_arr = np.array(chart)
# Create new dataframe from Numpy array to clearly display prior and 
# post results
pass_fail_chart = pd.DataFrame({'Prior Fail':chart_arr[:,0],
                                'Prior Pass':chart_arr[:,1]})
pass_fail_chart.index = pass_fail_chart.index.rename("")
pass_fail_chart = pass_fail_chart.rename(index={0:'Post Fail',1:'Post Pass'})


# The percentage of how many prior fails resulted in post fails
fail_fail_probability = (pass_fail_chart.loc['Post Fail', 'Prior Fail'] /
pass_fail_chart.loc['Post Fail', :].sum())
print(str(round(100*fail_fail_probability, 2)) + '%  of prior fails resulted in a post fail')

# The percentage of how many prior passes resulted in post fails
pass_fail_probability = (pass_fail_chart.loc['Post Pass', 'Prior Fail'] /
pass_fail_chart.loc['Post Pass', :].sum())
print(str(round(100*pass_fail_probability, 2))+ '% of prior passes resulted in a post fail')

# Calculate fines
data['fines'] = data[critical_columns].sum(axis=1) * 500
data['fines'] += data[serious_columns].sum(axis=1) * 250
data['fines'] += data[minor_columns].sum(axis=1) * 250

# Sort by date
data.sort_values('inspection_date', inplace=True)

# Calculate statistics for license groups
def get_fines(group):
    days = (group.iloc[-1].inspection_date - group.iloc[0].inspection_date).days + 1
    years = days / 365.25
    inspections = len(group)
    fines = group.fines.sum()
    yearly_fines = fines / math.ceil(years)
    risk = group.iloc[0].risk
    return pd.Series({
        'total_fines': fines,
        'yearly_fines': round(yearly_fines, 2),
        'risk_at_start':risk
    })

# Group by License and apply get_stats_1
fine_stats = data.groupby('license').apply(get_fines).reset_index()
print(fine_stats.head(20))
data = pd.merge(data, fine_stats, on='license')

# Create a new boolean variable: match
data['match'] = data.license.eq(data.license.shift())

# Calculate days since last inspection
data.loc[data['match'] == True, 'days_since_last'] = data['inspection_date'].diff().apply(lambda x: x/np.timedelta64(1, 'D')).fillna(0).astype('int64')  
data['days_since_last'] = data['days_since_last'].replace(np.nan, 0)
data['time_since_last']=(data['days_since_last']/365.25).round(2)

# Calculate days since first inspection
data['days_since_1st_inspection'] = data['inspection_date'].sub(data.groupby('license')['inspection_date'].transform('first'))
data['days_since_1st_inspection']=data['days_since_1st_inspection'].dt.days
data['time_since_1st_inspection']=(data['days_since_1st_inspection']/365.25).round(2)

# Calculate the number of previous inspections
data['previous_inspections']=data.sort_values('inspection_date').groupby('license').cumcount()

# Calculate the probability to pass the inspection
data['pass_chance'] = (data.groupby('license')['previous_results'].transform(lambda x: x.expanding().mean())).round(2)

# Calculate the probability to be found at least one critical violation
data['crit_found_chance'] = (data.groupby('license')['previous_critical_found'].transform(lambda x: x.expanding().mean())).round(2)

# Calculate the average of violations 
data['avg_critical'] = (data.groupby('license')['previous_critical_count'].transform(lambda x: x.expanding().mean())).round(2)
data['avg_serious'] = (data.groupby('license')['previous_serious_count'].transform(lambda x: x.expanding().mean())).round(2)
data['avg_minor'] = (data.groupby('license')['previous_minor_count'].transform(lambda x: x.expanding().mean())).round(2)

# Calculate the inspection's frequency
data['current_inspections']=data.sort_values('inspection_date').groupby('license').cumcount()+1
data['inspection_frequency']=(data['days_since_1st_inspection']/data['current_inspections']).round()


print(data.dtypes.sort_values())
print(data.shape)
print(data.columns.values)
print(data.isnull().sum().sort_values(ascending=False))
print(data.head(30))





# Business licenses have numbers on end preventing simple match
# so using street number instead
def get_street_number(address):
    return address.split()[0]

df_business['street_number'] = df_business.address.apply(get_street_number)
df_food['street_number'] = df_food.address.apply(get_street_number)

# Match based on dba_name and street_number
venue_matches = pd.merge(df_food, df_business, on=['dba_name', 'street_number'])

# Match based on License 
license_matches = pd.merge(df_food, df_business, on='license')

# Join matches, reset index, drop duplicates
matches = venue_matches.append(license_matches, sort=False)
matches.reset_index(drop=True, inplace=True)
matches.drop_duplicates(['inspection_id', 'id'], inplace=True)

# Restrict to matches where inspection falls within License_term_start_date and License_term_expiration_date 
matches = matches.loc[matches.inspection_date.between(matches.license_start_date, matches.license_expiration_date)]

# Keep only specific columns
matches=matches[['inspection_id','license_description', 
                 'license_start_date', 'license_expiration_date']]

# Merge the datasets
data = pd.merge(data, matches, on="inspection_id")
print(data.shape)

# Check for duplicates based on inspection_id
print('Data has', data.duplicated(subset=['inspection_id']).sum(), 'duplicated values')
data=data.drop_duplicates(subset=['inspection_id'])
print('Data has', data.duplicated(subset=['inspection_id']).sum(), 'duplicated values')
print(data.shape)

# Create new variable: age_of_inspection
def get_age_data(group):
    min_date = group.license_start_date.min()
    deltas = group.inspection_date - min_date
    group['age_of_inspection'] = deltas.apply(lambda x: x.days / 365.25)
    return group[['inspection_id', 'age_of_inspection']]

# Calculate and drop duplicates
age_data = data.groupby('license').apply(get_age_data).drop_duplicates()

# Merge in age_of_inspection
data = pd.merge(data, age_data, on='inspection_id')

# The distribution of the variable age_of_inspection
a=data['age_of_inspection'].plot(kind='hist')
a.set_title('Age at inspection')
a.set_ylabel('')
a.set_xlabel('')
plt.show()

# Create new variable: age_at_inspection 
data.loc[data['age_of_inspection'] >= 4, 'age_at_inspection'] = 1 
data.loc[data['age_of_inspection'] < 4, 'age_at_inspection'] = 0 

# Age at inspection over 4 years
value_counts = data['age_at_inspection'].value_counts()
value_counts.plot.bar(title = 'Age at inspection over 4 years')
plt.xticks(rotation=0)
plt.show()

print(data.dtypes.sort_values())
print(data.shape)
print(data.columns.values)
print(data.isnull().sum().sort_values(ascending=False))

# Create new variable: tobacco
data.loc[data['license_description'] == 'Tobacco', 'tobacco'] = 1 
data.loc[data['license_description'] != 'Tobacco', 'tobacco'] = 0 

# Create new variable: alcohol
data.loc[data['license_description'] == 'Consumption on Premises - Incidental Activity', 'alcohol'] = 1 
data.loc[data['license_description'] != 'Consumption on Premises - Incidental Activity', 'alcohol'] = 0 

# Tobacco
value_counts = data['tobacco'].value_counts()
value_counts.plot.bar(title = 'Tobacco')
plt.xticks(rotation=0)
plt.show()

# Alcohol
value_counts = data['alcohol'].value_counts()
value_counts.plot.bar(title = 'Alcohol')
plt.xticks(rotation=0)
plt.show()

# Drop the violation columns
data = data.drop(['v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 
              'v_11', 'v_12', 'v_13', 'v_14', 'v_15', 'v_16', 'v_17', 'v_18', 'v_19', 'v_20',
              'v_21', 'v_22', 'v_23', 'v_24', 'v_25', 'v_26', 'v_27', 'v_28', 'v_29', 'v_30',
              'v_31', 'v_32', 'v_33', 'v_34', 'v_35', 'v_36', 'v_37', 'v_38', 'v_39', 'v_40',
              'v_41', 'v_42', 'v_43', 'v_44', 'v_45', 'v_46', 'v_47', 'v_48', 'v_49', 'v_50',
              'v_51', 'v_52', 'v_53', 'v_54', 'v_55', 'v_56', 'v_57', 'v_58', 'v_59', 'v_60',
              'v_61', 'v_62', 'v_63', 'v_64', 'v_70',], axis = 1)

# Drop the p_violation columns
data = data.drop(['pv_1', 'pv_2', 'pv_3', 'pv_4', 'pv_5', 'pv_6', 'pv_7', 'pv_8', 'pv_9', 'pv_10', 
              'pv_11', 'pv_12', 'pv_13', 'pv_14', 'pv_15', 'pv_16', 'pv_17', 'pv_18', 'pv_19', 'pv_20',
              'pv_21', 'pv_22', 'pv_23', 'pv_24', 'pv_25', 'pv_26', 'pv_27', 'pv_28', 'pv_29', 'pv_30',
              'pv_31', 'pv_32', 'pv_33', 'pv_34', 'pv_35', 'pv_36', 'pv_37', 'pv_38', 'pv_39', 'pv_40',
              'pv_41', 'pv_42', 'pv_43', 'pv_44', 'pv_45', 'pv_46', 'pv_47', 'pv_48', 'pv_49', 'pv_50',
              'pv_51', 'pv_52', 'pv_53', 'pv_54', 'pv_55', 'pv_56', 'pv_57', 'pv_58', 'pv_59', 'pv_60',
              'pv_61', 'pv_62', 'pv_63', 'pv_64', 'pv_70',], axis = 1)

# Drop the data with no previous info 
data=data.dropna()

print(data.dtypes.sort_values())
print(data.shape)
print(data.columns.values)
print(data.isnull().sum().sort_values(ascending=False))
print(data.head(20))
print("------------------------------------------------------------")



# Criticals found per year
g1=data.groupby('year').critical_found.value_counts().unstack().plot.barh()
g1.set_title('Criticals found by year')
g1.set_ylabel('')
g1.set_xlabel('')
plt.show()

# Criticals found per month
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

g2=data.groupby('month').critical_found.value_counts().unstack().plot.barh()
g2.set_yticklabels(months)
g2.set_title('Criticals found by month')
g2.set_ylabel('')
g2.set_xlabel('')
plt.show()

# Criticals found - Results
plt.figure(figsize=(10,10))
plt.title('Criticals found - Results')
h=sns.heatmap(pd.crosstab([data.critical_found], [data.results]), square=True,
            cmap='Spectral', annot=True, fmt='.1f', linewidths=0.5, cbar=False)
plt.show()

# Garbage frequency per zip
i1=data.plot.scatter(x='zip', y='garbage_frequency', marker='o', figsize=(7,5))
i1.set_title('Garbage frequency by zip')
i1.set_ylabel('')
i1.set_xlabel('')
plt.show()

# Sanitation frequency per zip
i2=data.plot.scatter(x='zip', y='sanitation_frequency', marker='o', figsize=(7,5))
i2.set_title('Sanitation frequency by zip')
i2.set_ylabel('')
i2.set_xlabel('')
plt.show()

# Rodent frequency per zip
i3=data.plot.scatter(x='zip', y='rodent_frequency', marker='o', figsize=(7,5))
i3.set_title('Rodent frequency by zip')
i3.set_ylabel('')
i3.set_xlabel('')
plt.show()

#Set marker properties
markercolor = data['zip']

#Make Plotly figure
fig1 = go.Scatter3d(x=data['sanitation_frequency'],
                    y=data['garbage_frequency'],
                    z=data['rodent_frequency'],
                    marker=dict(color=markercolor,
                                opacity=1,
                                reversescale=True,
                                colorscale='Rainbow',
                                size=5),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title='Sanitation frequency'),
                                yaxis=dict( title='Garbage frequency'),
                                zaxis=dict(title='Rodent frequency')),)

#Plot and save html
plotly.offline.plot({'data': [fig1],
                     'layout': mylayout},
                     auto_open=False,
                     filename=('4DPlot.html'))

# Convert integers to strings
def currency(x, pos):
    """The two args are the value and tick position"""
    if x >= 1e6:
        s = '${:1.1f}M'.format(x*1e-6)
    else:
        s = '${:1.0f}K'.format(x*1e-3)
    return s

# Calculate the fines per month
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_fines = data.groupby('month').fines.sum() / data.fines.sum() 
# Plot bar chart
index = np.arange(len(month_fines))
fig, ax = plt.subplots()
ax.barh(index, month_fines)
ax.set_yticks(index)
ax.set_yticklabels(months)
ax.set_xticks([x*3/100 for x in range(5)])
ax.set_xticklabels(['%d%%'%(x*3) for x in range(5)])
ax.set_title('Fines by month')
ax.set_xlabel('Percent of all fines')
plt.show()

years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
year_fines = data.groupby('year').fines.sum() 
group_mean = np.mean(year_fines)
# Plot bar chart
index = np.arange(len(year_fines))
fig, ax = plt.subplots()
ax.barh(index, year_fines)
ax.set_yticks(index)
ax.set_yticklabels(years)
labels = ax.get_xticklabels()
plt.setp(labels) 
ax.axvline(group_mean, ls='--', color='r')
ax.set(title='Fines by year')
ax.xaxis.set_major_formatter(currency)
plt.show()


# Additional info
# The percentage of inspections in which they were found at least 1 violation
print('The percentage of inspections in which they were found at least 1 violation:', 
round((len(data.loc[data.violation_count >= 1]) / len(data))*100),'%')

# The average number of violations per inspection
print('The average number of violations per inspection:', round(data.violation_count.mean()))

# The average fine of each inspection
print('The average fine of each inspection:', round(data.fines.mean()),'$')

# The average yearly fine of each business
print('The average yearly fine of each business:', round(data.yearly_fines.mean()),'$')

# Τhe percentage to be found at least 1 critical violation
print('Τhe percentage to be found at least 1 critical violation:', round((len(data.loc[data.critical_found == 1]) / len(data))*100),'%')
print("------------------------------------------------------------")





# Keep only specific columns
many_data = data.loc[:, ['zip', 'previous_critical_count', 'previous_serious_count', 'previous_minor_count', 'previous_violation_count',
                         'previous_garbage_frequency','previous_sanitation_frequency', 'previous_rodent_frequency',
                         'risk_at_start', 'previous_month',
                         'previous_critical_found', 'previous_serious_found',
                         'time_since_1st_inspection', 'inspection_frequency', 'pass_chance',
                         'avg_critical', 'avg_serious', 'avg_minor', 
                         'crit_found_chance', 'time_since_last', 'previous_inspections',
                         'age_at_inspection', 'tobacco', 'alcohol', 
                         'previous_critical_ratio', 'previous_serious_ratio', 'previous_minor_ratio', 
                         'critical_found']]

print(many_data.dtypes.sort_values())
print(many_data.shape)
print(many_data.columns.values)
print(many_data.isnull().sum().sort_values(ascending=False))

# Correlation matrix
plt.figure(figsize=(30,30))
plt.title('Correlation matrix')
sns.heatmap(many_data.corr(), square=True, annot=True, linewidths=.5, cmap='Spectral')
plt.show()





# Machine learning for prediction





X, y = many_data.drop('critical_found', axis=1), many_data.critical_found
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# XGBClassifier method


# Set the parameters grid
paramGrid = {
         "learning_rate":[0.1],
         "n_estimators":[600],
         "max_depth": [5],
         "min_child_weight":[5],
         "gamma":[0.1],
         "subsample": [0.8],
         "colsample_bytree": [0.8],
         "colsample_bylevel": [0.8],
         "scale_pos_weight":[0.7]
            }
  
model_xgb = XGBClassifier(nthread=10)

cv = StratifiedKFold()

gridsearch = GridSearchCV(model_xgb, paramGrid, scoring='roc_auc', cv=cv, verbose=2)

fit_xgb = gridsearch.fit(X_train, y_train)

y_train_xgb_preds = fit_xgb.predict(X_train)
y_test_xgb_preds = fit_xgb.predict(X_test)

print('The best params:', fit_xgb.best_params_)
print('The roc_auc score:', (fit_xgb.best_score_)*100)
print('The train accuracy score:', accuracy_score(y_train,y_train_xgb_preds)*100)
print('The test accuracy score:', accuracy_score(y_test,y_test_xgb_preds)*100)

best = fit_xgb.best_estimator_

# Feature importance
fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
plot_importance(best, height=0.4, importance_type='gain', max_num_features=30, show_values=False, ax=ax)
plt.title('Feature importance (XGB)')
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, fit_xgb.predict(X_test))
f, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(conf_matrix, square=True, annot=True, fmt='d', linewidths=.5, cmap='magma_r', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# SHAP (SHapley Additive exPlanations) is a game theoretic approach 
# to explain the output of any machine learning model
explainer = shap.TreeExplainer(best)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type="bar")

for x in X.columns:
    shap.dependence_plot(x, shap_values, X, interaction_index=None)



# ROC curve + AUC score
ns_probs = [0 for _ in range(len(y_test))]
# Predict probabilities
xgb_probs = fit_xgb.predict_proba(X_test)

# Keep probabilities for the positive outcome only
xgb_probs = xgb_probs[:, 1]

# Calculate AUC
ns_auc = roc_auc_score(y_test, ns_probs)
xgb_auc = roc_auc_score(y_test, xgb_probs)

# Calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)

# Plot the roc curve for the model
plt.plot(xgb_fpr, xgb_tpr, color='blue', linewidth=3, label='XGBoost')
plt.plot(ns_fpr, ns_tpr, linestyle='dashed', color='green', linewidth=3, label='No Skill')
# Show title
plt.title('ROC curve')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show the legend
plt.legend()
# Show the plot
plt.show()

# Summarize scores
print('AUC score')
print('XGBoost: %.2f' % (xgb_auc))
print('No Skill: %.2f' % (ns_auc))



print(max(data['days_since_last']))

df_new = data.copy()
df_new["time_last_buckets"] = pd.qcut(df_new["days_since_last"], 500, labels=False, duplicates = "drop")

# Group by 'time_last_buckets' and find the average critical_found outcome per time_last_bucket
mean_deposit = df_new.groupby(["time_last_buckets"])["critical_found"].mean()

plt.plot(mean_deposit.index, mean_deposit.values)
plt.title("Mean % of critical found depending on time since last inspection")
plt.xlabel("Days since last inspection")
plt.ylabel("% Critical found")
plt.show()



# Add the prediction column
y_hats = fit_xgb.predict(X_test)
y_hats_df = pd.DataFrame(data = y_hats, columns = ['prediction'], index = X_test.index.copy())
results = pd.merge(data, y_hats_df, how = 'left', left_index = True, right_index = True)

results = results[results.prediction.isin([1])]
results.sort_values(by=['previous_critical_count', 'previous_minor_count'], ascending=False, inplace=True)
results=results.drop_duplicates(subset='license', keep='first')
print(results.head(15))



# The number of next inspections
print('This will take only a few seconds')
goodinput = False
while not goodinput:
    try:
        number = int(input('Enter the number of inspections: '))
        if number > 0:
            goodinput = True
            print('Thank you for your submission')
        else:
            print('Please enter a positive number')
    except ValueError:
        print('Please enter an integer nunber')


        
# Create a new dataframe with the next inspection details
final=results[:number]
print(final.shape)



# Where will the next inspections take place
long=final.longitude.mean()
lat=final.latitude.mean()
final_map=folium.Map([lat,long],zoom_start=8)

insp_distribution_map=plugins.MarkerCluster().add_to(final_map)
for lat,lon,label_1, label_2 in zip(final.latitude,final.longitude,final['dba_name'], final['address']):
    folium.Marker(location=[lat,lon],icon=None,popup=label_1 + label_2).add_to(insp_distribution_map)
final_map.add_child(insp_distribution_map)

final_map.save('next_inspections.html')
webbrowser.open('next_inspections.html')
