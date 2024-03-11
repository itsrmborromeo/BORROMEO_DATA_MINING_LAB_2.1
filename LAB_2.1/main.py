import pandas as pd
import numpy as np

df = pd.read_csv('data/titanic.csv')

df.head()

df.describe()

print(df.dtypes)
print(df.info())

len(df[df.Survived==0])/len(df)*100.0

df_grouped = df.groupby(by='Pclass')
print(df_grouped.Survived.count())
print(df_grouped.Survived.sum())
print(df_grouped.Survived.sum()/df_grouped.Survived.count())

df['age_range'] = pd.cut(df.Age, [0, 16, 65, 1e6], 3, labels=['child', 'adult', 'senior'])
df.age_range.describe()

df_grouped = df.groupby(by=['Pclass', 'age_range'])
print("Percentage of survivors in each group: ")
print(df_grouped.Survived.sum()/df_grouped.Survived.count() * 100)

for col in ['PassenderId', 'Name', 'Cabin', 'Ticket']:
    if col in df:
        del df[col]
       
df_grouped = df.groupby(by=['Pclass', 'SibSp'])


numeric_cols = df.select_dtypes(include=[np.number]).columns 
df_imputed = df_grouped[numeric_cols].transform(lambda grp: grp.fillna(grp.median()))

non_numeric_cols = ['Pclass', 'SibSp', 'Sex', 'Embarked']

for col in non_numeric_cols:
    df_imputed[col] = df[col]

print(df_imputed.info())