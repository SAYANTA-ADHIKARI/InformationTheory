import numpy as np

def marginal1(pmf, i):
    if i > 3:
        print("Wrong index!!")
        return
    pmf = pmf.reshape(-1, 10 ** (i+1))
    scale = 10 ** i
    x = [pmf[:, l * scale : (l+1) * scale].sum() for l in range(10)]
    return np.array(x)

def marginal2(pmf, i, j):
    scale1, scale2 = (i, j) if i > j else (j, i)
    pmf = pmf.reshape(-1, 10 ** (scale1+1))
    x = [] ; scale = 10 ** scale1
    for l in range(10):
        x.append(marginal1(pmf[:, l * scale : (l+1) * scale], scale2))
    if i > j:
        return np.array(x)
    else:
        return np.array(x).T


def mutual_information(pmf, i, j):
    if i == j:
        raise RuntimeError()
    p_i = marginal1(pmf, i)
    p_j = marginal1(pmf, j)
    p_ij = marginal2(pmf, i, j)

    mi = 0
    for l in range(10):
        for k in range(10):
            val = p_ij[l, k] / (p_i[l] * p_j[k])
            if val == 0: continue
            else:
                mi += p_ij[l, k] * np.log2 (val)
    return mi