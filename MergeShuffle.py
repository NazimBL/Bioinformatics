data = data2 = ""

# Reading data from file1
with open('gata3_positive.fa') as fp:
	data = fp.read()

# Reading data from file2
with open('gata3_negative.fa') as fp:
	data2 = fp.read()

data += "\n"
data += data2

with open ('gata3_merged', 'w') as fp:
	fp.write(data)

// shuffle	
import random
with open('gata3_merged','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('gata3_final','w') as target:
    for _, line in data:
        target.write( line )

#delete first line of a file
with open('positive_final.csv', 'r') as fin:
    data = fin.read().splitlines(True)
with open('positive_finall.csv', 'w') as fout:
    fout.writelines(data[1:])
