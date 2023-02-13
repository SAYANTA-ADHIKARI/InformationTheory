import numpy as np
from timeit import default_timer as timer

# import the function from your submission "Serial_101.py"
from Serial_101 import mutual_information
# generating a probability mass function ‘pmf’
# pmf = np.random.rand(10000)
# pmf = pmf / sum(pmf)
pmf = np.load('Sample_pmf.npy')
pmf1 = np.load('Sample_pmf.npy')
# computing the mutual information between
# two random variables X_i and X_j, i not equal to j
# i, j belong to the set {0,1,2,3}
i = 2
j = 0
start = timer()
MI = mutual_information(pmf,i,j)
end = timer()
print(pmf1.sum())
print(MI)
# print(end - start)