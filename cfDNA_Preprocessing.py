import pandas as pd

#loop this
output = open("positive_data_table", "w")

for i in range(0,46):
    df=pd.read_table("coverage_"+str(i)+".tab")
    # Drop single column by Index
    df2=df.drop(df.columns[[0, 1, 2]], axis = 1)

    df2.to_csv('out_'+str(i)+'.csv', index=False)
    i+=1

#loop this
for j in range(0,46):
    with open('out_'+str(j)+'.csv', 'r') as f:          # Read lines separately
        for readline in f:
            line_strip = readline.strip()
            output.write(line_strip)
            output.write(',')

        output.write('\n')
        f.close()
output.close()
