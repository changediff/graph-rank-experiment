def cosine_sim (vec1, vec2):
    import numpy as np
    from numpy import linalg as la

    inA = np.mat(vec1)
    inB = np.mat(vec2)
    num = float(inA * inB.T) #若为行向量: A * B.T
    donom = la.norm(inA) * la.norm(inB) ##余弦值 
    return 0.5 + 0.5*(num / donom) # 归一化
    #关于归一化：因为余弦值的范围是 [-1,+1] ，相似度计算时一般需要把值归一化到 [0,1]

a = [1,1,1]
b = [2,-3,-1]

print(cosine_sim(a,b))