import random

result = []
x = 0.0
output = open('result.txt', 'w')
nums = [random.gauss(30, 2) for _ in range(120)]
for num in nums:
    result.append((('{0:.1f}'.format(x)) + ';'))
    output.write((('{0:.1f}'.format(x)) + ';'))
    #print(('{0:.1f}'.format(x)))
    x = x + num

print(result)



output.close()