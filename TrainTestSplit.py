import random

fin = open('gata3_final', 'rb')
f70out = open("gata3_train", 'wb')
f30out = open("gata3_test", 'wb')


for line in fin:
  r = random.random()
  if (0.0 <=  r <= 0.70):
    f70out.write(line)
  else:
    f30out.write(line)
fin.close()
f70out.close()
f30out.close()
