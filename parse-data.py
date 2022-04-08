#input file
fin = open("dataset.txt", "rt")
#output file to write the result to
fout = open("out.txt", "wt")
#for each line in the input file
for line in fin:
	#read replace the string and write to output file
    line = line.replace('b,', '0,')
    line = line.replace('x,', '1,')
    line = line.replace('o,', '2,')
    fout.write(line)
#close input and output files
fin.close()
fout.close()