import numpy as np

def computeNumericalGradient(J, theta):
  """ Compute numgrad = computeNumericalGradient(J, theta)

  theta: a vector of parameters
  J: a function that outputs a real-number and the gradient.
  Calling y = J(theta)[0] will return the function value at theta. 
  """

  # Initialize numgrad with zeros
  numgrad = np.zeros(theta.size)

  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Implement numerical gradient checking, and return the result in numgrad.  
  # You should write code so that numgrad(i) is (the numerical approximation to) the 
  # partial derivative of J with respect to the i-th input argument, evaluated at theta.  
  # I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
  # respect to theta(i).
  #                
  # Hint: You will probably want to compute the elements of numgrad one at a time. 
  eps = 1.0e-4
  eps2 = 2*eps
  for i in range(theta.size):
    theta_p = theta.copy()
    theta_n = theta.copy()
    theta_p[i] = theta[i] + eps
    theta_n[i] = theta[i] - eps

    numgrad[i] = (J(theta_p)[0] - J(theta_n)[0]) / eps2





  ## ---------------------------------------------------------------

  return numgrad
