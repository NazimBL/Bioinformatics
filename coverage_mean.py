import pandas as pd
import os

#loop this
df=pd.read_table("coverage_1.tab")
    # Drop single column by Index
df2=df.drop(df.columns[[0, 1, 2]], axis = 1)
df2.to_csv('out.csv', index=False)

given_file = open('out.csv', 'r')

lines = given_file.readlines()
sum = 0

for line in lines:
    for c in line:
        if c.isdigit() == True:
            sum = sum + int(c)

print(sum)

given_file.close()

os.rename('out.csv','out_'+str(sum)+'.csv')
