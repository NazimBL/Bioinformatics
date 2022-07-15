import csv

positive_path = "Esbr_positive"


fasta = open("Esbr_positive_prep", "w")

with open(positive_path, 'r') as f:          # Read lines separately
    reader = csv.reader(f, delimiter=',')
    for line in enumerate(reader):
        data=str(line[1]).upper()
        data=data.replace("[","")
        data=data.replace("]", "")
        data=data.replace("'","")
        fasta.write(str(data))
        fasta.write("\t")
        fasta.write("1")
        fasta.write("\n")
f.close()
fasta.close()
