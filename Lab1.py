def fpitch(p):
    return 440 * 2**((p-69)/12)

def computeP(fs, N, K, p):
    cutoff_freq = (fpitch(p-0.5), fpitch(p+0.5))
    f_coef = lambda k: k * fs / N
    
    P = []
    for k in range(K):
        coef = f_coef(k)
        if coef >= cutoff_freq[0] and coef < cutoff_freq[1]:
            P.append(k)
    
    return P
