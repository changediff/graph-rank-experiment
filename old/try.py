import math

vec1 = [1,2,3]
vec2 = [4,5,6]

tmp = list(map(lambda x: abs(x[0]-x[1]), zip(vec1, vec2)))

print(list(tmp))

distance = math.sqrt(sum(map(lambda x: x*x, tmp)))

print(distance)