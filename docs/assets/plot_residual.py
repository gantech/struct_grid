import matplotlib.pyplot as plt
import pandas as pd 

jacobi = pd.read_csv('jacobi_resid.txt')
adi = pd.read_csv('adi_resid.txt')
cg = pd.read_csv('cg_resid.txt')
mg = pd.read_csv('mg_resid.txt')

fig = plt.figure()
plt.semilogy(jacobi.iloc[:,1], label='Jacobi', color='C0')
plt.semilogy(adi.iloc[:,1], label='ADI', color='C1')
plt.semilogy(cg.iloc[:,1], label='Conjugate Gradient', color='C2')
plt.semilogy(mg.iloc[:,1], label='Geometric Multigrid', color='C3')
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.legend(loc=0)
plt.grid()
plt.savefig('residual_plot.png', dpi=300)
plt.close(fig)