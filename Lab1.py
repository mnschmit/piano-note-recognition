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

def pitchbasedSTFT(X):
    # transform the frequencies into pitches from a normal STFT X
    import numpy as np
    Y = np.abs(X) ** 2

    ### first the frequencies, then the time! ###
    K, N = X.shape

    Y_LF = np.zeros((128, N))
    for n in range(N):
        for p in range(128):
            Y_LF[p][n] = sum(map(lambda k: Y[k][n], computeP(fs, window_length, K, p)))
    
    return Y_LF
