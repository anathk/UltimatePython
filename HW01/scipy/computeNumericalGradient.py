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
  # This value can be changed to smaller value.
  eps = 1.0e-4

  for i in range(theta.size):
    theta_previous = theta.copy()
    theta_next = theta.copy()
    theta_previous[i] = theta[i] + eps
    theta_next[i] = theta[i] - eps

    numgrad[i] = 0.5 * (J(theta_previous)[0] - J(theta_next)[0]) / eps
  ## ---------------------------------------------------------------

  return numgrad
