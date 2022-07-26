import os
import pandas as pd

for filename in os.listdir("new/"):

    fasta = open(filename+"_out.tab", "w")

    with open("new/"+filename, 'r') as f:          # Read lines separately
        Lines = f.readlines()
        for line in Lines:

            from re import search
            substring = "chrY"

            if search(substring, str(line)): fasta.write(str(line))
            else: continue
        
    f.close()
    fasta.close()

#loop this
    df=pd.read_table(filename+"_out.tab")
    # Drop single column by Index
    print(df.head())
    df2=df.drop(df.columns[[0, 1, 2]], axis = 1)
    df2.to_csv(filename+"_out.csv", index=False)

    given_file = open(filename+"_out.csv", 'r')

    lines = given_file.readlines()
    sum = 0

    for line in lines:
        for c in line:
            if c.isdigit() == True:
                sum = sum + int(c)

    print(sum)

    given_file.close()
    os.rename(filename+"_out.csv",filename+str(sum)+'.csv')
