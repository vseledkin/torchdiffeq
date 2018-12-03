import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
  from torchdiffeq import odeint_adjoint as odeint
else:
  from torchdiffeq import odeint


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      t0=200,
                      noise_std=.2,
                      to_tensor=False):
  """Parametric formula for 2d spiral is `r = a + b * theta`.

  Args:
    nspiral: number of spirals, i.e. batch dimension number
    other args same as above
  Returns: 
    Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
    second element is noisy observations of size (nspiral, nsample, 2),
    third element is timestamps of size (nspiral, ntotal),
    and fourth element is timestamps of size (nspiral, nsample)
  """
  # sample a and b from Gaussian
  a, b = npr.randn(nspiral, 1), npr.randn(nspiral, 1) + 1.
  thetas = np.linspace(start=0, stop=6 * np.pi, num=ntotal)  # (ntotal,)
  rs = a + b * thetas[None, :]  # (nspiral, ntotal)
  xs, ys = rs * np.cos(thetas), rs * np.sin(thetas)
  # create batch
  orig_traj = np.stack((xs, ys), axis=2)  # (nspiral, ntotal, 2)
  orig_ts = thetas
  samp_traj = orig_traj[:, t0:t0 + nsample, :].copy()
  samp_traj += npr.randn(*samp_traj.shape) * noise_std
  samp_ts = thetas[t0:t0 + nsample]

  if to_tensor:
    orig_traj = torch.from_numpy(orig_traj).float()
    samp_traj = torch.from_numpy(samp_traj).float()
    orig_ts = torch.from_numpy(orig_ts).float()
    samp_ts = torch.from_numpy(samp_ts).float()

  return orig_traj, samp_traj, orig_ts, samp_ts


class LatentODEfunc(nn.Module):

  def __init__(self, latent_dim=4, nhidden=20):
    super(LatentODEfunc, self).__init__()
    self.elu = nn.ELU(inplace=True)
    self.fc1 = nn.Linear(latent_dim, nhidden)
    self.fc2 = nn.Linear(nhidden, nhidden)
    self.fc3 = nn.Linear(nhidden, latent_dim)
    self.nfe = 0

  def forward(self, t, x):
    self.nfe += 1
    out = self.fc1(x)
    out = self.elu(out)
    out = self.fc2(out)
    out = self.elu(out)
    out = self.fc3(out)
    return out


class Decoder(nn.Module):

  def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
    super(Decoder, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.fc1 = nn.Linear(latent_dim, nhidden)
    self.fc2 = nn.Linear(nhidden, obs_dim)

  def forward(self, z):
    out = self.fc1(z)
    out = self.relu(out)
    out = self.fc2(out)
    return out


class RecognitionRNN(nn.Module):

  def __init__(self, latent_dim=4, obs_dim=2, nhidden=25):
    super(RecognitionRNN, self).__init__()

    self.nhidden = nhidden
    self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
    self.h2o = nn.Linear(nhidden, latent_dim * 2)

  def forward(self, x, h):
    combined = torch.cat((x, h), dim=1)
    h = torch.tanh(self.i2h(combined))
    out = self.h2o(h)  # mean and logvar for approx. posterior
    return out, h

  def initHidden(self):
    return torch.zeros(1, self.nhidden)


class RunningAverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, momentum=0.99):
    self.momentum = momentum
    self.reset()

  def reset(self):
    self.val = None
    self.avg = 0

  def update(self, val):
    if self.val is None:
      self.avg = val
    else:
      self.avg = self.avg * self.momentum + val * (1 - self.momentum)
    self.val = val


def log_normal_pdf(x, mean, logvar):
  const = torch.log(torch.from_numpy(np.array([2. * np.pi])).float())
  return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


if __name__ == '__main__':
  # constants
  latent_dim = 4
  nhidden = 25
  rnn_nhidden = 25
  obs_dim = 2
  noise_std = .2
  nspiral = 1  # TODO: fit more than one dynamics

  # generate toy data
  orig_traj, samp_traj, orig_ts, samp_ts = generate_spiral2d(
      nspiral=nspiral, noise_std=noise_std, to_tensor=True)

  if torch.cuda.is_available():
    device = torch.device('cuda:' + str(args.gpu))
  else:
    device = torch.device('cpu')

  # model
  # TODO: compose into single `nn.Module`
  func = LatentODEfunc(latent_dim=latent_dim, nhidden=nhidden).to(device)
  dec = Decoder(
      latent_dim=latent_dim, obs_dim=obs_dim, nhidden=nhidden).to(device)
  rec = RecognitionRNN(
      latent_dim=latent_dim, obs_dim=obs_dim, nhidden=rnn_nhidden).to(device)
  params = (list(func.parameters()) +
            list(dec.parameters()) + list(rec.parameters()))
  optimizer = optim.RMSprop(params, lr=args.lr)
  for itr in range(1, args.niters + 1):
    optimizer.zero_grad()
    # backward in time to infer q(z_0)
    h = rec.initHidden()
    for t in reversed(range(samp_traj.size(1))):
      obs = samp_traj[:, t, :].to(device)
      out, h = rec.forward(obs, h)
    qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
    z0 = torch.randn(qz0_mean.size()) * torch.exp(.5 * qz0_logvar) + qz0_mean

    # forward in time and solve ode for reconstructions
    pred_x = []
    pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
    for t in range(pred_z.size(1)):
      zt = pred_z[:, t, :]
      xt = dec(zt)
      pred_x.append(xt)
    pred_x = torch.stack(pred_x, dim=1)

    # debug
    if itr > 300:
      orig_traj_ = orig_traj.numpy()
      samp_traj_ = samp_traj.numpy()
      xs = pred_x.detach().numpy()
      plt.figure()
      plt.plot(xs[0, :, 0], xs[0, :, 1], 'r', label='learned traj')
      plt.plot(
          orig_traj_[0, :, 0], orig_traj_[0, :, 1], 'g', label='true traj')
      plt.scatter(
          samp_traj_[0, :, 0], samp_traj_[0, :, 1], s=3, label='sampled data')
      plt.legend()
      plt.show()

    # compute loss
    noise_std_ = torch.zeros(pred_x.size()) + noise_std  # to Tensor
    noise_logvar = 2. * torch.log(noise_std_)
    logpx = log_normal_pdf(samp_traj, pred_x, noise_logvar).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(z0.size())
    logpz0 = log_normal_pdf(z0, pz0_mean, pz0_logvar)
    logqz0 = log_normal_pdf(z0, qz0_mean, qz0_logvar)
    mc_kl = torch.sum(logqz0 - logpz0, dim=1)
    loss = torch.mean(-logpx + mc_kl, dim=0)
    print('Iter: {}, elbo: {:.4f}'.format(itr, -loss.cpu().detach().numpy()))
    loss.backward()
    optimizer.step()
  print("Training complete.")

  # sample latent traj from prior
  with torch.no_grad():
    xs = []
    z0 = torch.randn(1, latent_dim)
    zs = odeint(func, z0, samp_ts).permute(1, 0, 2)
    for t in range(zs.size(1)):
      zt = zs[:, t, :].to(device)
      xt = dec(zt)
      xs.append(xt)
    xs = torch.stack(xs, dim=1)

  xs = xs.cpu().numpy()
  orig_traj = orig_traj.cpu().numpy()
  samp_traj = samp_traj.cpu().numpy()

  # visualize
  plt.figure()
  plt.plot(orig_traj[0, :, 0], orig_traj[0, :, 1], 'g', label='true traj')
  plt.plot(xs[0, :, 0], xs[0, :, 1], 'r', label='learned traj')
  plt.scatter(
      samp_traj[0, :, 0], samp_traj[0, :, 1], s=3, label='sampled data')
  plt.legend()
  plt.show()
