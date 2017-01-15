import cmath
import math

print(8*cmath.cos(math.radians(40)) - 6*cmath.cos(math.radians(80)))
print(-8*cmath.sin(math.radians(40)) + 6*cmath.sin(math.radians(80)))

#Add two 2D vectors in the format of (magnitude, degree)
def add_vector(vector1, vector2):
    return (vector1[0]*cmath.cos(math.radians(vector1[1])) + vector2[0]*cmath.cos(math.radians(vector2[1])),
            vector1[0]*cmath.sin(math.radians(vector1[1])) + vector2[0]*cmath.sin(math.radians(vector2[1])))

print(add_vector([8, 320], [6, 100]))