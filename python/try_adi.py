import numpy as np
import matplotlib.pyplot as plt

class LaplaceHeat:

  def __init__(self, nx=10, ny=30):
    self.kc = 0.001
    self.nx = nx
    self.ny = ny
    self.dx = 1.0/nx
    self.dy = 3.0/ny
    self.J = np.zeros((self.nx,self.ny,5))
    self.R = np.zeros((self.nx,self.ny))
    self.T = np.ones((self.nx,self.ny)) * 300.0
    self.deltaT = np.zeros((self.nx,self.ny))

  def compute_r_j(self):
    for i in range(1,self.nx-1):
      for j in range(1,self.ny-1):
        y = (0.5 + j) * self.dy
        self.R[i,j] = self.kc * (self.T[i+1,j] + self.T[i-1,j] + self.T[i,j+1] + self.T[i,j-1] - 4.0 * self.T[i,j]) # - (2 + 2.0 * y/9.0) * self.dx * self.dy )
        self.J[i,j,0] = -4.0 * self.kc #i,j
        self.J[i,j,1] = self.kc #i-1,j
        self.J[i,j,2] = self.kc #i+1,j
        self.J[i,j,3] = self.kc #i,j-1
        self.J[i,j,4] = self.kc #i,j+1

    #Bottom left - Dirichlet, Dirichlet
    x = 0.5 * self.dx
    y = 0.5 * self.dy
    phi_bc_left = 300.0 + (y*y*y/27.0)
    phi_bc_bot = 300.0 + x * x
    self.R[0,0] = self.kc * ( -(9.0 * self.T[0,0] - self.T[1,0] - 8.0 * phi_bc_left)/3.0 - (9.0 * self.T[0,0] - self.T[0,1] - 8.0 * phi_bc_bot)/3.0 + self.T[1,0] + self.T[0,1] - 2.0 * self.T[0,0]) # - (2.0 + 2.0 * y/9.0) * self.dx * self.dy )
    self.J[0,0,0] = -8.0 * self.kc #i,j
    self.J[0,0,1] = 0 #i-1,j
    self.J[0,0,2] = 4.0 * self.kc / 3.0 #i+1,j
    self.J[0,0,3] = 0 #i,j-1
    self.J[0,0,4] = 4.0 * self.kc / 3.0 #i,j+1

    x = 0.0
    for j in range(1,self.ny-1):
      y = (0.5 + j) * self.dy
      phi_bc_left = 300.0 + (y*y*y/27.0)
      #Left boundary - Dirichlet
      self.R[0,j] = self.kc * ( -(9.0 * self.T[0,j] - self.T[1,j] - 8.0 * phi_bc_left)/3.0 + self.T[1,j] - 3.0 * self.T[0,j] + self.T[0,j+1] + self.T[0,j-1]) # - (2.0 + 2.0 * y/9.0) * self.dx * self.dy )
      self.J[0,j,0] = -6.0 * self.kc #i,j
      self.J[0,j,1] = 0.0 #i-1,j
      self.J[0,j,2] = 4.0 * self.kc/3.0 #i+1,j
      self.J[0,j,3] = self.kc #i,j-1
      self.J[0,j,4] = self.kc #i,j+1
      #Right boundary - Neumann 
      self.R[self.nx-1,j] = self.kc * ( self.T[self.nx-1,j+1] + self.T[self.nx-1,j-1] + self.T[self.nx-2,j] - 3.0 * self.T[self.nx-1,j] + 2.0 * self.dy) # - (2.0 + 2.0 * y/9.0) * self.dx * self.dy )
      self.J[self.nx-1,j,0] = -3.0 * self.kc #i,j
      self.J[self.nx-1,j,1] = self.kc #i-1,j
      self.J[self.nx-1,j,2] = 0 #i+1,j
      self.J[self.nx-1,j,3] = self.kc #i,j-1
      self.J[self.nx-1,j,4] = self.kc #i,j+1

    #Top left - Dirichlet, Neumann
    y = (0.5 + (self.ny-1) ) * self.dy
    phi_bc_left = 300.0 + (y*y*y/27.0)
    self.R[0,self.ny-1] = self.kc * ( -(9.0 * self.T[0,self.ny-1] - self.T[1,self.ny-1] - 8.0 * phi_bc_left)/3.0 + self.T[1,self.ny-1] + self.T[0,self.ny-2] - 2.0 * self.T[0,self.ny-1] + self.dx) # - (2.0 + 2.0 * y/9.0) * self.dx * self.dy )
    self.J[0,self.ny-1,0] = -5.0 * self.kc #i,j
    self.J[0,self.ny-1,1] = 0 #i-1,j
    self.J[0,self.ny-1,2] = 4.0 * self.kc / 3.0 #i+1,j
    self.J[0,self.ny-1,3] = self.kc #i,j-1
    self.J[0,self.ny-1,4] = 0 #i,j+1

    #Bottom right - Neumann, Dirichlet
    y = 0.5 * self.dy
    x = (0.5 + self.nx-1) * self.dx
    phi_bc_bot = 300.0 + x * x
    self.R[self.nx-1,0] = self.kc * ( -(9.0 * self.T[self.nx-1,0] - self.T[self.nx-1,1] - 8.0 * phi_bc_bot)/3.0 + self.T[self.nx-2,0] + self.T[self.nx-1,1] - 2.0 * self.T[self.nx-1,0] + 2.0 * self.dy) # - (2.0 + 2.0 * y/9.0) * self.dx * self.dy )
    self.J[self.nx-1,0,0] = -5.0 * self.kc #i,j
    self.J[self.nx-1,0,1] = self.kc #i-1,j
    self.J[self.nx-1,0,2] = 0 #i+1,j
    self.J[self.nx-1,0,3] = 0 #i,j-1
    self.J[self.nx-1,0,4] = 4.0 * self.kc / 3.0 #i,j+1

    for i in range(1,self.nx-1):
      x = (0.5 + i) * self.dx
      phi_bc_bot = 300.0 + x * x
      y = 0.5 * self.dy
      #Bottom boundary - Dirichlet
      self.R[i,0] = self.kc * ( self.T[i+1,0] + self.T[i-1,0] + self.T[i,1] - 3.0 * self.T[i,0] - (9.0 * self.T[i,0] - self.T[i,1] - 8.0 * phi_bc_bot)/3.0) # - (2.0 + 2.0 * y/9.0) * self.dx * self.dy )
      self.J[i,0,0] = -6.0 * self.kc #i,j
      self.J[i,0,1] = self.kc #i+1,j
      self.J[i,0,2] = self.kc #i-1,j
      self.J[i,0,3] = 0 #i,j-1
      self.J[i,0,4] = 4.0 * self.kc / 3.0 #i,j+1
      #Top boundary - Neumann
      y = (0.5 + (self.ny-1) ) * self.dy
      self.R[i,self.ny-1] = self.kc * ( self.T[i+1,self.ny-1] + self.T[i-1,self.ny-1] + self.T[i,self.ny-2] - 3.0 * self.T[i,self.ny-1] + self.dx) # - (2.0 + 2.0 * y/9.0) * self.dx * self.dy )
      self.J[i,self.ny-1,0] = -3.0 * self.kc #i,j
      self.J[i,self.ny-1,1] = self.kc #i+1,j
      self.J[i,self.ny-1,2] = self.kc #i-1,j
      self.J[i,self.ny-1,3] = self.kc #i,j-1
      self.J[i,self.ny-1,4] = 0 #i,j+1

    #Top right - Neumann, Neumann
    y = (0.5 + (self.ny-1) ) * self.dy
    self.R[self.nx-1,self.ny-1] = self.kc * ( self.T[self.nx-1,self.ny-2] + self.T[self.nx-2,self.ny-1] - 2.0 * self.T[self.nx-1, self.ny-1] + self.dx + 2.0 * self.dy) # - (2.0 + 2.0 * y/9.0) * self.dx * self.dy )
    self.J[self.nx-1,self.ny-1,0] = -2.0 * self.kc #i,j
    self.J[self.nx-1,self.ny-1,1] = self.kc #i-1,j
    self.J[self.nx-1,self.ny-1,2] = 0 #i+1,j
    self.J[self.nx-1,self.ny-1,3] = self.kc #i,j-1
    self.J[self.nx-1,self.ny-1,4] = 0 #i,j+1

    return np.linalg.norm(self.R)



  def adi_x(self):

    for j in range(self.ny):
      a = self.J[:,j,1]
      b = self.J[:,j,0]
      c = self.J[:,j,2]
      d = -self.R[:,j]
      self.deltaT[:,j] = self.thomas(a,b,c,d)
    self.T[:,:] += self.deltaT[:,:]


  def adi_y(self):

    for i in range(self.nx):
      a = self.J[i,:,3]
      b = self.J[i,:,0]
      c = self.J[i,:,4]
      d = -self.R[i,:]
      self.deltaT[i,:] = self.thomas(a,b,c,d)
    self.T[:,:] += self.deltaT[:,:]


  def thomas(self,a,b,c,d):

    n = np.size(d)
    x = np.zeros(n)
    for i in range(1,n):
      w = a[i] / b[i-1]
      b[i] -= w * c[i-1]
      d[i] -= w * d[i-1]

    x[n-1] = d[n-1]/b[n-1]
    for i in range(n-2,-1,-1):
      x[i] = (d[i] - c[i] * x[i+1]) / b[i]

    return x


  def adi_solve(self, nloops):
    resid = np.zeros(nloops)
    for k in range(nloops):
      resid[k] = self.compute_r_j()
      self.adi_x()
      self.compute_r_j()
      self.adi_y()

    x = np.arange(0.5 * self.dx, 1.0, self.dx)
    y = np.arange(0.5 * self.dy, 3.0, self.dy)
    X, Y = np.meshgrid(x,y)
    self.Tref = np.zeros((self.nx, self.ny))
    for i in range(self.nx):
      x = (0.5 + i) * self.dx
      for j in range(self.ny):
        y = (0.5 + j) * self.dy
        self.Tref[i,j] = 300.0 + x*x + (y*y*y/27.0)

    print(resid)
    print(np.linalg.norm(self.T - self.Tref))
    fig = plt.figure()
    cs = plt.contourf(X,Y, (self.T-self.Tref).transpose())
    fig.colorbar(cs)
    plt.show()

    return resid

  def plot_resid(self):
    x = np.arange(0.5 * self.dx, 1.0, self.dx)
    y = np.arange(0.5 * self.dy, 3.0, self.dy)
    X, Y = np.meshgrid(x,y)
    for i in range(self.nx):
      x = (0.5 + i) * self.dx
      for j in range(self.ny):
        y = (0.5 + j) * self.dy
        self.T[i,j] = 300.0 + x*x + (y*y*y/27.0)        
    self.compute_r_j()
    Rref = self.kc * (2.0 + 2.0 * Y/9.0)

    self.R = self.R/(self.dx * self.dy)
    print(Rref)
    print(self.R.transpose())
    fig = plt.figure()  
    cs = plt.contourf(X,Y, (Rref - self.R.transpose())/Rref )
    print(np.max(np.abs(Rref - self.R.transpose())))
    
    fig.colorbar(cs)
    plt.show()


if __name__=="__main__":

    # l = LaplaceHeat(nx=10, ny=30)
    # resid1 = l.adi_solve(2000)

    # l = LaplaceHeat(nx=20, ny=60)
    # resid2 = l.adi_solve(2000)

    #l = LaplaceHeat(nx=40, ny=120)
    #resid3 = l.adi_solve(100)

    #fig = plt.figure()
    #plt.semilogy(resid1)
    #plt.semilogy(resid2)
    #plt.semilogy(resid3)
    #plt.grid()
    #plt.show()

    l = LaplaceHeat(nx = 32, ny = 96)   
    l.plot_resid()
    l = LaplaceHeat(nx = 256, ny = 768)   
    l.plot_resid()    