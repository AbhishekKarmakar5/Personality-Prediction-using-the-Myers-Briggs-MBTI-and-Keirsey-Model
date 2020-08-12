import re
import numpy as np
import collections
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
#https://github.com/jesusgarcia2/Personality

"""
The database we are working with classifies people into 16 distinct personality types showing their last 50 tweets, 
separated by "|||". Our goal will be to create new columns based on the content of the tweets, in order to create a 
predictive model. 
"""
df = pd.read_csv('mbti_1.csv')
print(df.head(10))
print("*"*300)
print(df.info())
print("*"*100)
print(df.groupby('type').describe())

"""
We can thus see that there are no null inputs, which means there is no need for cleaning the data.
The first idea that pops up is checking if the words per tweet of each person shows us some information. 
For that reason, we can create a new column as shown below.
"""
print("*"*300)
df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
print(df.head())

# Exploratory data analysis
"""
We may use it for one reason or for another, but one thing we can do is printing a violin plot.
At the end I did not use it at all, but it is always nice to have the ability do some visual analysis 
for further investigations.
"""
"""
plt.figure(figsize=(15,10))
sns.violinplot(x='type', y='words_per_comment', data=df, inner=None, color='lightgray')
sns.stripplot(x='type', y='words_per_comment', data=df, size=4, jitter=True)
plt.ylabel('Words per comment')
plt.show()
"""

"""
Creating new columns showing the amount of questionmarks per comment, exclamations or other types will be useful 
later on, as we will see. 
This are the examples I came up with, but here is where creativity comes into play.
We can also perform joint plots, pair plots and heat maps to explore relationship between data, just for fun.
"""
df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)
df['music_per_comment'] = df['posts'].apply(lambda x: x.count('music')/50)
df['question_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)
df['img_per_comment'] = df['posts'].apply(lambda x: x.count('jpg')/50)
df['excl_per_comment'] = df['posts'].apply(lambda x: x.count('!')/50)
df['ellipsis_per_comment'] = df['posts'].apply(lambda x: x.count('...')/50)

"""plt.figure(figsize=(15,10))
sns.jointplot(x='words_per_comment', y='ellipsis_per_comment', data=df, kind='kde')
plt.show()"""

#print(df.head())

# So it seems there's a large correlation between words per comment ant the ellipsis the user types per commment
"""
i = df['type'].unique()
k = 0
for m in range(0,2):
    for n in range(0,6):
        df_2 = df[df['type'] == i[k]]
        sns.jointplot(x='words_per_comment', y='ellipsis_per_comment', data=df_2, kind="hex")
        plt.title(i[k])
        plt.show()
        k+=1
"""
i = df['type'].unique()
k = 0
TypeArray = []
PearArray=[]
for m in range(0,2):
    for n in range(0,6):
        df_2 = df[df['type'] == i[k]]
        pearsoncoef1=np.corrcoef(x=df_2['words_per_comment'], y=df_2['ellipsis_per_comment'])
        pear=pearsoncoef1[1][0]
        print(pear)
        TypeArray.append(i[k])
        PearArray.append(pear)
        k+=1


TypeArray = [x for _,x in sorted(zip(PearArray,TypeArray))]
PearArray = sorted(PearArray, reverse=True)
print(PearArray)
print(TypeArray)
#plt.scatter(TypeArray, PearArray)
#plt.show()


"""

-------------------------------------------------------------------Data preprocessing---------------------------------

To get a further insight on our dataset, we can first create 4 new columns dividing the people by 
introversion/extroversion, intuition/sensing, and so on.
When it comes to performing machine learning, trying to distinguish between two categories is much easier than 
distinguishing between 16 categories. We will check that later on. Dividing the data in 4 small groups will 
perhaps be more useful when it comes to accuracy.
"""
map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
df['I-E'] = df['type'].astype(str).str[0]
df['I-E'] = df['I-E'].map(map1)
df['N-S'] = df['type'].astype(str).str[1]
df['N-S'] = df['N-S'].map(map2)
df['T-F'] = df['type'].astype(str).str[2]
df['T-F'] = df['T-F'].map(map3)
df['J-P'] = df['type'].astype(str).str[3]
df['J-P'] = df['J-P'].map(map4)
print(df.head(10))

#------------------------------------------------------------------------------------------------------------
# Building machine learning algorithms

X = df.drop(['type','posts','I-E','N-S','T-F','J-P'], axis=1).values
y = df['type'].values

print(y.shape)
print(X.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.1, random_state=5)

sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
sgd.score(X_train, y_train)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
print(round(acc_sgd,2,), "%")
print("Accuracy Score of SDG:",accuracy_score(y_test, Y_pred))

# ----------------------------------------  Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")
print("Accuracy Score of Random Forest:",accuracy_score(y_test, Y_prediction))

# --------------------------------------- Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print(round(acc_log,2,), "%")
print("Accuracy Score of LogisticRegression:",accuracy_score(y_test, Y_pred))
# ----------------------------------------- KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print(round(acc_knn,2,), "%")
print("Accuracy Score of KNN:",accuracy_score(y_test, Y_pred))

"""
Our simple model is only able to classify people with a 24% of right guesses, which is not too much.
Now we will perform machine learning with the introverted/extroverted column, and we'll see if our model is 
able to classify if someone is introverted or extroverted with a higher precision.
"""
print("------------------------------------------------------------------")
XX = df.drop(['type','posts','I-E'], axis=1).values
yy = df['I-E'].values

print(yy.shape)
print(XX.shape)

XX_train,XX_test,yy_train,yy_test=train_test_split(XX,yy,test_size = 0.1, random_state=5)

sgdd = SGDClassifier(max_iter=5, tol=None)
sgdd.fit(XX_train, yy_train)
Y_predd = sgdd.predict(XX_test)
sgdd.score(XX_train, yy_train)
acc_sgdd = round(sgdd.score(XX_train, yy_train) * 100, 2)
print("The accuracy of SGDClassifier")
print(round(acc_sgdd,2,), "%")
print("confusion_matrix for SGDClassifier")
print(confusion_matrix(yy_test,Y_predd))
print("Accuracy Score :",accuracy_score(yy_test, Y_predd))
print(classification_report(yy_test,Y_predd))

#Random Forest
random_forestt = RandomForestClassifier(n_estimators=100)
random_forestt.fit(XX_train, yy_train)

Y_predictionn = random_forestt.predict(XX_test)

random_forestt.score(XX_train, yy_train)
acc_random_forestt = round(random_forestt.score(XX_train, yy_train) * 100, 2)
print("The accuracy of Random Forest classifier")
print(round(acc_random_forestt,2,), "%")
print("confusion_matrix for Random Forest")
print(confusion_matrix(yy_test,Y_predictionn))
print("Accuracy Score :",accuracy_score(yy_test, Y_predictionn))
print(classification_report(yy_test,Y_predictionn))

# Logistic Regression
logregg = LogisticRegression()
logregg.fit(XX_train, yy_train)

Y_predd = logregg.predict(XX_test)

acc_logg = round(logregg.score(XX_train, yy_train) * 100, 2)
print("The accuracy of Logistic Regression")
print(round(acc_logg,2,), "%")

print("confusion_matrix for Logistic Regression")
print(confusion_matrix(yy_test,Y_predd))
print("Accuracy Score :",accuracy_score(yy_test, Y_predd))
print(classification_report(yy_test,Y_predd))

# KNN
knnn = KNeighborsClassifier(n_neighbors = 3)
knnn.fit(XX_train, yy_train)

Y_predd = knnn.predict(XX_test)

acc_knnn = round(knnn.score(XX_train, yy_train) * 100, 2)
print("The accuracy of KNN")
print(round(acc_knnn,2,), "%")

print("confusion_matrix for KNN")
print(confusion_matrix(yy_test,Y_predd))
print("Accuracy Score :",accuracy_score(yy_test, Y_predd))
print(classification_report(yy_test,Y_predd))

error_rate = []
for i in range(1,40):
	knn = KNeighborsClassifier(n_neighbors = i)
	knn.fit(XX_train, yy_train)
	pred_i = knn.predict(XX_test)
	error_rate.append(np.mean(pred_i!=yy_test))

print("------------------>>",error_rate)
"""
plt.figure(figsize=(15,10))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
						markerfacecolor='red',markersize=10)
plt.title("Error Rate vs K Value")
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()"""

"""
So we see our model has an accuracy of 77%, not bad for such a simple model! Let's see what else we can do!
"""

new_column=[]
for z in range(len(df['posts'])):
    prov=df['posts'][z]
    prov2= re.sub(r'[“€â.|,?!)(1234567890:/-]', '', prov)
    prov3 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', prov)
    prov4 = re.sub(r'[|||)(?.,:1234567890!]',' ',prov3)
    prov5 = re.sub(' +',' ', prov4)
    prov6 = prov5.split(" ")
    counter = Counter(prov6)
    counter2 = counter.most_common(1)[0][0]
    new_column.append(counter2)
df['most_used_word'] = new_column
print(df.head())
print(df['most_used_word'].unique())

#---------------------nlp----------------------
df['length'] = df['posts'].apply(len)
print("*"*100)
print(df.head(10))

#df['length'].plot.hist(bins=100,alpha=0.5)
fig,ax = plt.subplots(1,2)
ax[0].hist(df['length'],bins='auto')
ax[0].set_title("histogram of result")

#df.hist(column = 'length', by = 'type', bins=60, figsize=(15,10) )
ax[1]=df['length'].plot(kind='hist',by='type',bins=60)
ax[1].set_title("Length column")
#plt.show()



df.assign(dummy = 1).groupby(
  ['dummy','type']
).size().to_frame().unstack().plot(kind='bar',stacked=True,legend=False)

plt.title('Number of persons having such personality')

# other it'll show up as 'dummy' 
plt.xlabel('type')

# disable ticks in the x axis
plt.xticks([])

# fix the legend
current_handles, _ = plt.gca().get_legend_handles_labels()
reversed_handles = reversed(current_handles)

labels = reversed(df['type'].unique())

#plt.legend(reversed_handles,labels,loc='lower right')
#plt.show()

#http://queirozf.com/entries/pandas-dataframe-plot-examples-with-matplotlib-pyplot

map1 = {"ESTP":1,"ISTP":1,"ESFP":1,"ISFP":1}
map2 = {"ESTJ":1,"ISTJ":1,"ESFJ":1,"ISFJ":1}
map3 = {"ENFJ":1,"INFJ":1,"ENFP":1,"INFP":1}
map4 = {"ENTJ":1,"INTJ":1,"ENTP":1,"INTP":1}

df['Artisian'] = df['type'].astype(str)
df['Artisian'] = df['Artisian'].map(map1)
df['Artisian'] = df['Artisian'].fillna(0)
df['Guardian'] = df['type'].astype(str)
df['Guardian'] = df['Guardian'].map(map2)
df['Guardian'] = df['Guardian'].fillna(0)
df['Idealist'] = df['type'].astype(str)
df['Idealist'] = df['Idealist'].map(map3)
df['Idealist'] = df['Idealist'].fillna(0)
df['Rational'] = df['type'].astype(str)
df['Rational'] = df['Rational'].map(map4)
df['Rational'] = df['Rational'].fillna(0)
print(df.head(10))
print(type(df['I-E'][0]))
print(type(df['Guardian'][0]))
df.Artisian = df.Artisian.astype('int64')
df.Guardian = df.Guardian.astype('int64')
df.Idealist = df.Idealist.astype('int64')
df.Rational = df.Rational.astype('int64')
print(type(df['I-E'][0]))
print(type(df['Guardian'][0]))
print(df.head(10))
print("------------------------------Keirsey Model------------------------------------")
print("*"*200)
XXX = df.drop(['type','posts','Guardian'], axis=1).values
yyy = df['Guardian'].values

print(yyy.shape)
print(XXX.shape)

XXX_train,XXX_test,yyy_train,yyy_test=train_test_split(XX,yy,test_size = 0.33, random_state=5)

sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(XXX_train, yyy_train)
YY_predd = sgd.predict(XXX_test)
sgdd.score(XX_train, yy_train)
acc_sgdd = round(sgdd.score(XXX_train, yyy_train) * 100, 2)
print("The accuracy of SGDClassifier")
print(round(acc_sgdd,2,), "%")
print("confusion_matrix for SGDClassifier")
print(confusion_matrix(yyy_test,YY_predd))
print("Accuracy Score :",accuracy_score(yyy_test, YY_predd))
print(classification_report(yyy_test,YY_predd))

#Random Forest
random_forestt = RandomForestClassifier(n_estimators=100)
random_forestt.fit(XXX_train, yyy_train)

YY_predictionn = random_forestt.predict(XXX_test)

random_forestt.score(XXX_train, yyy_train)
acc_random_forestt = round(random_forestt.score(XXX_train, yyy_train) * 100, 2)
print("The accuracy of Random Forest classifier")
print(round(acc_random_forestt,2,), "%")
print("confusion_matrix for Random Forest")
print(confusion_matrix(yyy_test,YY_predictionn))
print("Accuracy Score :",accuracy_score(yyy_test, YY_predictionn))
print(classification_report(yyy_test,YY_predictionn))

# Logistic Regression
logregg = LogisticRegression()
logregg.fit(XXX_train, yyy_train)

YY_predd = logregg.predict(XXX_test)

acc_logg = round(logregg.score(XXX_train, yyy_train) * 100, 2)
print("The accuracy of Logistic Regression")
print(round(acc_logg,2,), "%")

print("confusion_matrix for Logistic Regression")
print(confusion_matrix(yyy_test,YY_predd))
print("Accuracy Score :",accuracy_score(yyy_test, YY_predd))
print(classification_report(yyy_test,YY_predd))

# KNN
knnn = KNeighborsClassifier(n_neighbors = 3)
knnn.fit(XXX_train, yyy_train)

YY_predd = knnn.predict(XXX_test)

acc_knnn = round(knnn.score(XXX_train, yyy_train) * 100, 2)
print("The accuracy of KNN")
print(round(acc_knnn,2,), "%")

print("confusion_matrix for KNN")
print(confusion_matrix(yyy_test,YY_predd))
print("Accuracy Score :",accuracy_score(yyy_test, YY_predd))
print(classification_report(yyy_test,YY_predd))
