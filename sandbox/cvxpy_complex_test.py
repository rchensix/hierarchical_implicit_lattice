# Import packages.
import cvxpy as cp
import numpy as np


# Define and solve the CVXPY problem.
n = 10
x = cp.Variable(n, complex=True)
prob = cp.Problem(cp.Minimize(cp.norm(x)))
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)