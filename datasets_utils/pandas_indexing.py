import pandas as pd

df = pd.DataFrame(
    {
    'name':['Jane', 'Nick', 'Aaron', 'Penelope'],
    'age':[30, 2, 12, 4],
    'food':['Steak', 'Lamb', 'Mango', 'Apple'],
    'vegan':[False, False, True, True],
    'score':[4.6, 8.3, 9.0, 3.3],
    },
)

all_names = df['name'].unique()
person = df.loc[df['name']=='Jane']

for name in all_names:
    person = df.loc[df['name']==name]
    print(int(person['age']))
    print(person['food'].values[0])
    print(person['vegan'].bool())
    print(float(person['score']))
    print("==========")