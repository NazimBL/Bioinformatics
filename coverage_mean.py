import os

import pandas as pd

positive_path = "f_healthy_coverage_1.tab"


fasta = open("f_healthy_coverage_1_out.tab", "w")

with open(positive_path, 'r') as f:          # Read lines separately
    Lines = f.readlines()
    for line in Lines:

        from re import search
        substring = "chrY"

        if search(substring, str(line)): fasta.write(str(line))
        else: continue
        
f.close()
fasta.close()

#loop this
df=pd.read_table("f_healthy_coverage_1_out.tab")
    # Drop single column by Index
print(df.head())
df2=df.drop(df.columns[[0, 1, 2]], axis = 1)
df2.to_csv('hout.csv', index=False)

given_file = open('hout.csv', 'r')

lines = given_file.readlines()
sum = 0

for line in lines:
    for c in line:
        if c.isdigit() == True:
            sum = sum + int(c)

print(sum)

given_file.close()
os.rename('hout.csv','hout_'+str(sum)+'.csv')
