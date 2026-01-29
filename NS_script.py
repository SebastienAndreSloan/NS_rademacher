import torch
import numpy as np
from torch.autograd import grad
from tqdm import tqdm

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iterations = 20000
d_Ns = [30] # These are the widths
N_rs = [3, 5, 15] # Number of collocation points, the real value will be cubed.
second_layer_weight_mean = 0
second_layer_weight_std = 1
tanh3act = True # Set to True to use tanh^3 activation, False for tanh.

# parameters of the PDE
min_xyt, max_xyt = 0, 1
nu = 0.01
rho = 1

# training parameters
test_rate = 100 # number of test loss computations
lambda_0 = 1
lambda_1 = 0.3
lr = 1e-3
lambda_s = 0.5
loss_choice = torch.nn.HuberLoss()


# Collocation points generator
def coll_gen(N_r, N_0):
    x_coll_r, y_coll_r, t_coll_r = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, N_r),
          torch.linspace(min_xyt, max_xyt, N_r),
          torch.linspace(0, max_xyt - min_xyt, N_r),
          indexing = "xy"
      )
    coll_r = torch.stack([x_coll_r,y_coll_r,t_coll_r],dim=-1).reshape(-1,3).to(dev)

    x_coll_0, y_coll_0 = torch.meshgrid(
            torch.linspace(min_xyt, max_xyt, N_0),
            torch.linspace(min_xyt, max_xyt, N_0),
            indexing = "xy"
        )
    coll_0 = torch.stack([x_coll_0, y_coll_0],dim=-1).reshape(-1,2).to(dev)

    return coll_r, coll_0

# ------------ Other Collocation Data ------------------
N_0 = 20 # squared (Initial condition)
N_t = 60 # cubed (Test Volume)
N_0_t = 60 # squared (Test IC)

torch.manual_seed(0)
np.random.seed(0)
x_coll_test, y_coll_test, t_coll_test= torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, N_t),
          torch.linspace(min_xyt, max_xyt, N_t),
          torch.linspace(0, max_xyt - min_xyt, N_t),
          indexing = "xy"
      )
coll_test = torch.stack([x_coll_test,y_coll_test,t_coll_test],dim=-1).reshape(-1,3).to(dev)

x_coll_0_test, y_coll_0_test = torch.meshgrid(
          torch.linspace(min_xyt, max_xyt, N_0_t),
          torch.linspace(min_xyt, max_xyt, N_0_t),
          indexing = "xy"
      )
coll_0_test = torch.stack([x_coll_0_test,y_coll_0_test],dim=-1).reshape(-1,2).to(dev)

# initial condition
def g_in(x,y):
    u = -1 * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
    v = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    p = rho / -4 * (torch.cos(2*torch.pi*x)+torch.cos(2*torch.pi*y))
    return torch.stack((u,v, p),dim=-1)

# Two different activation functions avaliable
class TanhCubed(torch.nn.Module):
    def forward(self, x):
        return torch.tanh(x) ** 3

class TwoLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)  # First layer (trainable)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)  # Second layer (frozen)
        self.activation = TanhCubed()

    def forward(self, x):
        if tanh3act:
           x = self.activation(self.layer1(x))
        else:
            x = torch.tanh(self.layer1(x))
        x = self.layer2(x)  # This layer will not be trained
        return x

def train(N, N_r):
  train_loss = np.zeros(iterations)
  test_loss = np.zeros(iterations // test_rate)
  optimizer = torch.optim.AdamW(N.parameters(), lr=lr)
  # for i in range(iterations):
  for i in tqdm(range(iterations)):
      coll_r, coll_0 = coll_gen(N_r, N_0)
      x_r, y_r, t_r = coll_r[:,0], coll_r[:,1], coll_r[:,2]
      x_r.requires_grad_(True)
      y_r.requires_grad_(True)
      t_r.requires_grad_(True)

      x_0, y_0 = coll_0[:,0],coll_0[:,1]
      x_0.requires_grad_(True)
      y_0.requires_grad_(True)
      
      optimizer.zero_grad()

      # Compute the partial derivatives using automatic differentiation
      out = N(torch.stack((x_r, y_r, t_r),dim=-1))
      u=out[...,0]
      v=out[...,1]
      p=out[...,2]

      u_x = grad(u, x_r, torch.ones_like(u), create_graph = True)[0].squeeze() # du/dx
      u_y = grad(u, y_r, torch.ones_like(u), create_graph = True)[0].squeeze() # du/dy
      u_t = grad(u, t_r, torch.ones_like(u), create_graph = True)[0].squeeze()  # du/dt
      v_x = grad(v, x_r, torch.ones_like(v), create_graph = True)[0].squeeze() # dv/dx
      v_y = grad(v, y_r, torch.ones_like(v), create_graph = True)[0].squeeze() # dv/dy
      v_t = grad(v, t_r, torch.ones_like(u), create_graph = True)[0].squeeze()  # dv/dt
      p_x = grad(p, x_r, torch.ones_like(p), create_graph = True)[0].squeeze() # dp/dx
      p_y = grad(p, y_r, torch.ones_like(p), create_graph = True)[0].squeeze() # dp/dy

      u_xx = grad(u_x, x_r, torch.ones_like(u_x), create_graph = True)[0].squeeze() # d2u/dx2
      u_yy = grad(u_y, y_r, torch.ones_like(u_y), create_graph = True)[0].squeeze() # d2u/dy2
      v_xx = grad(v_x, x_r, torch.ones_like(v_x), create_graph = True)[0].squeeze() # d2v/dx2
      v_yy = grad(v_y, y_r, torch.ones_like(v_y), create_graph = True)[0].squeeze() # d2v/dy2

      x_mom = u_t + (u * u_x) + (v * u_y) + ((1 / rho) * p_x) - (nu * (u_xx + u_yy)) # x momentum
      y_mom = v_t + (u * v_x) + (v * v_y) + ((1 / rho) * p_y) - (nu * (v_xx + v_yy)) # y momentum
      cont = u_x + v_y # continuity

      x_mom_loss = loss_choice(x_mom, torch.zeros_like(x_mom)) # x momentum loss
      y_mom_loss = loss_choice(y_mom, torch.zeros_like(y_mom)) # y momentum loss
      cont_loss = loss_choice(cont, torch.zeros_like(cont)) # continuity loss

      u0 = N(torch.stack((x_0,y_0,torch.zeros_like(x_0)),dim=-1))
      initial_loss = loss_choice(u0, g_in(x_0,y_0))

      # Compute the PDE loss
      pde_loss = x_mom_loss + y_mom_loss + lambda_0 * cont_loss

      # supervision_loss_train = supervision_loss_choice(N(X_train), Y_train)
      # Compute the total loss and perform a gradient step
      train = pde_loss + lambda_1*initial_loss
      train_loss[i] = train
      train.backward()
      optimizer.step()

      if i % test_rate == 0:
        for p in N.parameters():
            p.requires_grad_(False)
        N.eval()
        x_test, y_test, t_test = coll_test[:,0], coll_test[:,1], coll_test[:,2]
        x_test.requires_grad_(True)
        y_test.requires_grad_(True)
        t_test.requires_grad_(True)

        x_0_test, y_0_test = coll_0_test[:,0],coll_0_test[:,1]
        x_0_test.requires_grad_(True)
        y_0_test.requires_grad_(True)

        u0_test = N(torch.stack((x_0_test,y_0_test,torch.zeros_like(x_0_test)),dim=-1))
        initial_loss_test = loss_choice(u0_test, g_in(x_0_test,y_0_test))

        out_test = N(torch.stack((x_test, y_test, t_test),dim=-1))
        u_test=out_test[...,0]
        v_test=out_test[...,1]
        p_test=out_test[...,2]

        u_x_test = grad(u_test, x_test, torch.ones_like(u_test), create_graph = True)[0].squeeze() # du/dx
        u_y_test = grad(u_test, y_test, torch.ones_like(u_test), create_graph = True)[0].squeeze() # du/dy
        u_t_test = grad(u_test, t_test, torch.ones_like(u_test), create_graph = True)[0].squeeze()  # du/dt
        v_x_test = grad(v_test, x_test, torch.ones_like(v_test), create_graph = True)[0].squeeze() # dv/dx
        v_y_test = grad(v_test, y_test, torch.ones_like(v_test), create_graph = True)[0].squeeze() # dv/dy
        v_t_test = grad(v_test, t_test, torch.ones_like(u_test), create_graph = True)[0].squeeze()  # dv/dt
        p_x_test = grad(p_test, x_test, torch.ones_like(p_test), create_graph = True)[0].squeeze() # dp/dx
        p_y_test = grad(p_test, y_test, torch.ones_like(p_test), create_graph = True)[0].squeeze() # dp/dy

        u_xx_test = grad(u_x_test, x_test, torch.ones_like(u_x_test), create_graph = True)[0].squeeze() # d2u/dx2
        u_yy_test = grad(u_y_test, y_test, torch.ones_like(u_y_test), create_graph = True)[0].squeeze() # d2u/dy2
        v_xx_test = grad(v_x_test, x_test, torch.ones_like(v_x_test), create_graph = True)[0].squeeze() # d2v/dx2
        v_yy_test = grad(v_y_test, y_test, torch.ones_like(v_y_test), create_graph = True)[0].squeeze() # d2v/dy2

        x_mom_test = u_t_test + (u_test * u_x_test) + (v_test * u_y_test) + ((1 / rho) * p_x_test) - (nu * (u_xx_test + u_yy_test)) # x momentum
        y_mom_test = v_t_test + (u_test * v_x_test) + (v_test * v_y_test) + ((1 / rho) * p_y_test) - (nu * (v_xx_test + v_yy_test)) # y momentum
        cont_test = u_x_test + v_y_test # continuity

        x_mom_loss_test = loss_choice(x_mom_test, torch.zeros_like(x_mom_test)) # x momentum loss
        y_mom_loss_test = loss_choice(y_mom_test, torch.zeros_like(y_mom_test)) # y momentum loss
        cont_loss_test = loss_choice(cont_test, torch.zeros_like(cont_test)) # continuity loss

        # Compute the PDE loss
        pde_loss_test = x_mom_loss_test + y_mom_loss_test + lambda_0 * cont_loss_test

        test = pde_loss_test + lambda_1*initial_loss_test
        test_loss[i // test_rate] = test
        N.train()
        for p in N.parameters():
            p.requires_grad_(True)

  return train_loss, test_loss

num_dN = len(d_Ns)
num_Nr = len(N_rs)

train_errors = np.zeros((num_Nr,num_dN,iterations))
test_errors = np.zeros((num_Nr,num_dN,iterations // 100))
net_dict = {}

# ------------ Actual Training ------------------
result = np.zeros((num_Nr,num_dN),dtype=bool)
for j in range(num_Nr):
  for k in range(num_dN):
    NN = 0
    NN = TwoLayerNN(3,d_Ns[k],3).to(dev)
    for param in NN.layer2.parameters():
        param.requires_grad = False
    for name, param in NN.named_parameters():
      if "bias" in name:
          param.data.fill_(0)
          param.requires_grad = False
    with torch.no_grad():
        NN.layer2.weight.copy_(torch.normal(second_layer_weight_mean, second_layer_weight_std, size=NN.layer2.weight.shape) / np.sqrt(d_Ns[k]))  # /sqrtd normalization
    print(f"Starting net number {j} with width {d_Ns[k]} at Nr = {N_rs[j]}")
    train_errors[j,k,:], test_errors[j,k,:] = train(NN, N_rs[j])
    net_dict[f"net_Nr_{N_rs[j]}"] = NN.state_dict()
  print("Succefully done net number", j)
torch.save(net_dict, f"weights_NS_tanh_dNs_{d_Ns}_Nrs_{N_rs}_i_{iterations}.pt")
np.savez(f"result_NS_tanh_dNs_{d_Ns}_Nrs_{N_rs}_i_{iterations}",train=train_errors,test=test_errors)