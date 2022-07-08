#use bedtools getfasta beforhand
# open file in read mode
fn = open('gata3_alone.fa', 'r')

# open other file in write mode
fn1 = open('gata3_alone_test', 'w')

# read the content of the file line by line
cont = fn.readlines()
type(cont)
for i in range(0, len(cont)):
    if (i % 2  != 0):
        fn1.write(cont[i].upper())
    else:
        pass

# close the file
fn1.close()
