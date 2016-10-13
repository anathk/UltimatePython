import numpy

print(numpy.random.rand(4,4))

randMat = numpy.mat(numpy.random.rand(4,4))

invRandMat = randMat.I

myEye = randMat*invRandMat

print(myEye - numpy.eye(4))