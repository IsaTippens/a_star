import random
import sys

# python generate_squares.py <n> <output_path.txt>
n = int(sys.argv[1])

# generate list of number from 1 to n^2
numbers = list(range(1, n**2 + 1))
# shuffle the list
random.shuffle(numbers)

output_file = sys.argv[2]
with open(output_file, 'w') as f:
    f.write(str(n) + '\n')
    for i in range(n**2):
        f.write(str(numbers[i]) + ' ')

