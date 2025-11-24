from functools import reduce 

a = [ [0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2]]
print(reduce(lambda count, l: count + len(l) - 1, a, 0))

print(a[1:-1])
