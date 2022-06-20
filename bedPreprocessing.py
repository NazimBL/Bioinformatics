import csv

positive_path = "GATA3_positive_examples.csv"
negative_path = "GATA3_negative_examples.csv"

fasta = open("gata3_negative.fa", "w")
fasta.write("sequence")
fasta.write("\t")
fasta.write("label")
fasta.write("\n")

with open(negative_path, 'r') as f:          # Read lines separately
    reader = csv.reader(f, delimiter=',')
    for line in enumerate(reader):
        data=str(line[1]).upper()
        data=data.replace(",", "")
        data = data.replace("0", "")
        data = data.replace("'", "")
        data=data.replace("[", "")
        data=data.replace("]", "")
        data = data.replace(" ", "")
        fasta.write(data)
        fasta.write("\t")
        fasta.write("0")
        fasta.write("\n")
f.close()
fasta.close()
