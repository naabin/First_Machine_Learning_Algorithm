import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# pd.set_option('display.max_columns', 8)

df1 = pd.read_csv('../data/general.csv')
df2 = pd.read_csv('../data/prenatal.csv')
df3 = pd.read_csv('../data/sports.csv')

df2 = df2.set_axis(df1.columns, axis=1)
df3 = df3.set_axis(df1.columns, axis=1)
df4 = pd.concat([df1, df2, df3], ignore_index=True)
df4 = df4.drop('Unnamed: 0', axis=1)
df4.dropna(axis=0, inplace=True, how='all')
df4['gender'] = df4['gender'].map({'female': 'f', 'woman': 'f', 'man': 'm', 'male': 'm'})
prenatal_hospital = df4[df4['hospital'] == 'prenatal']
prenatal_hospital['gender'] = prenatal_hospital['gender'].fillna('f')
df4[df4['hospital'] == 'prenatal'] = prenatal_hospital
cols = ['bmi', 'diagnosis', 'blood_test', 'ecg', 'ultrasound', 'mri', 'xray', 'children', 'months']
df4[cols] = df4[cols].fillna(0)
q1 = pd.pivot_table(df4, index='hospital', aggfunc='count').idxmax()[0]
g_h = df4.loc[df4.hospital == 'general']
q2 = round(len(g_h[g_h['diagnosis'] == 'stomach']) / len(g_h), 3)
s_h = df4.loc[df4.hospital == 'sports']
q3 = round(len(s_h[s_h['diagnosis'] == 'dislocation'])/len(s_h), 3)
q4 = g_h['age'].median() - s_h['age'].median()
q5 = df4.loc[df4.blood_test == 't'].groupby('hospital')['blood_test'].count()
print(f'The answer to the 1st question is {q1}')
print(f'The answer to the 2nd question is {q2}')
print(f'The answer to the 3rd question is {q3}')
print(f'The answer to the 4th question is {abs(q4)}')
print(f'The answer to the 5th question is {q5.index[q5.argmax()]}, {q5.values[q5.argmax()]} blood tests')
print(df4['age'].tolist())
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.hist(df4['age'].tolist(), bins='sqrt')
ax1.set_title('Age distribution by hospitalization')
p = df4.groupby('diagnosis')['diagnosis'].count()
labels = p.index.tolist()
vals = p.values.tolist()
ax2.pie(vals, labels=labels)
ax2.set_title('Proportion of diagnosis')
sns.violinplot(df4, x='hospital', y='height')
plt.show()
print('The answer to the 1st question: 15-35')
print('The answer to the 2nd question: pregnancy')
print('The answer to 3rd question: It\'s because the age group between 35-55 happens to hospitalize '
      'more than any group for both male and female')


