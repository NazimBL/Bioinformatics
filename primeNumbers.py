Y_file = open("y.txt", 'w')
Z_file = open("z.txt", 'w')

y_index=0

for y_i in range(1, 10000):
       if y_i > 1:
         #if number is divisible by any other number than 1, and the number itself, its NOT a prime number
        for i in range(2, y_i):
           if (y_i % i) == 0:
               break
        else:
           Y_file.write(str(y_i))
           Y_file.write('\n')
           #keep track of i's that forms z
           y_index=y_index+1
           
           #if the index is not a prime number, write it to the z file
           for j in range(2, y_index):
              if (y_index % j) == 0:
                 Z_file.write(str(y_index))
                 Z_file.write('\n')
                 break


Y_file.close
Z_file.close            