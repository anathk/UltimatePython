import numpy as np

def computeNumericalGradient(J, theta):
  """Compute numgrad = computeNumericalGradient(J, theta)

  theta: a vector of parameters
  J: a function that outputs a real-number and the gradient.
  Calling y = J(theta)[0] will return the function value at theta. 
  """

  # Initialize numgrad with zeros
  numgrad = np.zeros(theta.size)

  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Implement numerical gradient checking, and return the result in numgrad.  
  # (See Section 2.3 of the lecture notes.)
  # You should write code so that numgrad(i) is (the numerical approximation to) the 
  # partial derivative of J with respect to the i-th input argument, evaluated at theta.  
  # I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
  # respect to theta(i).
  #                
  # Hint: You will probably want to compute the elements of numgrad one at a time. 

  ## ---------------------------------------------------------------

  return numgrad
