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


  # Need to convert theta to float array, otherwise value will be rounded to integers.
  numgrad = numgrad.astype(float)
  theta = theta.astype(float)

  # delta(y) / delta(x) = f(x + step) - f(x - step) / 2*step.
  step = 0.0001

  for i in range(theta.size):
    theta_previous = theta.copy()
    theta_next = theta.copy()
    theta_previous[i] = theta_previous[i] + step
    theta_next[i] = theta_next[i] - step

    numgrad[i] = (J(theta_previous)[0] - J(theta_next)[0]) / (2 * step)
  ## ---------------------------------------------------------------

  return numgrad
