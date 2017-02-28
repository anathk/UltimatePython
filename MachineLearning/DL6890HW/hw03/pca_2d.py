import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

##================================================================
## Step 0: Load data
#  We have provided the code to load data from pcaData.txt into x.
#  x is a 2 * 45 matrix, where the kth column x(:,k) corresponds to
#  the kth data point. You do not need to change the code below.

x = np.loadtxt('pcaData.txt')
plt.figure(1)
plt.scatter(x[0, :], x[1, :])
plt.title('Raw data')
plt.savefig('figure1.jpg')
plt.show()

##================================================================
## Step 1a: Implement PCA to obtain U 
#  Implement PCA to obtain the rotation matrix U, which is the eigenbasis
#  sigma.

# -------------------- YOUR CODE HERE --------------------

sigma = x.dot(x.T) / x.shape[1]
u, s, v = np.linalg.svd(sigma)


# -------------------------------------------------------- 

plt.figure(2)
plt.plot([0, u[0, 0]], [0, u[1, 0]])
plt.plot([0, u[0, 1]], [0,u[1, 1]])
plt.scatter(x[0, :], x[1, :])
plt.title('PCA basis')
plt.savefig('figure2.jpg')
plt.show()

##================================================================
## Step 1b: Compute xRot, the projection on to the eigenbasis
#  Now, compute xRot by projecting the data on to the basis defined
#  by U. Visualize the points by performing a scatter plot.

# -------------------- YOUR CODE HERE -------------------- 

xRot = u.T.dot(x)

# -------------------------------------------------------- 

# Visualise the covariance matrix. You should see a line across the
# diagonal against a blue background.
plt.figure(3)
plt.scatter(xRot[0, :], xRot[1, :])
plt.title('xRot')
plt.savefig('figure3.jpg')
plt.show()

##================================================================
## Step 2: Reduce the number of dimensions from 2 to 1. 
#  Compute xRot again (this time projecting to 1 dimension).
#  Then, compute xHat by projecting the xRot back onto the original axes 
#  to see the effect of dimension reduction

# -------------------- YOUR CODE HERE -------------------- 

xHat = u.dot(np.resize(xRot[0], x.shape))

# -------------------------------------------------------- 

plt.figure(4)
plt.scatter(xHat[0, :], xHat[1, :])
plt.title('xHat')
plt.savefig('figure4.jpg')
plt.show()


##================================================================
## Step 3: PCA Whitening
#  Compute xPCAWhite and plot the results.

epsilon = 1e-5

# -------------------- YOUR CODE HERE -------------------- 

xPCAWhite = np.diag(1 / (s + epsilon)).dot(xRot)

# --------------------------------------------------------

plt.figure(5)
plt.scatter(xPCAWhite[0, :], xPCAWhite[1, :])
plt.title('xPCAWhite')
plt.savefig('figure5.jpg')
plt.show()

##================================================================
## Step 3: ZCA Whitening
#  Compute xZCAWhite and plot the results.

# -------------------- YOUR CODE HERE -------------------- 

xZCAWhite = u.dot(xPCAWhite)

# -------------------------------------------------------- 

plt.figure(6)
plt.scatter(xZCAWhite[0, :], xZCAWhite[1, :])
plt.title('xZCAWhite')
plt.savefig('figure6.jpg')
plt.show()

## Congratulations! When you have reached this point, you are done!
#  You can now move onto the next PCA exercise. :)
