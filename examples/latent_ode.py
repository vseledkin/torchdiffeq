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
import torch.nn.functional as F


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
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      to_tensor=False,
                      a=0.,
                      b=1.,
                      device='cpu'):
  """Parametric formula for 2d spiral is `r = a + b * theta`.

  Args:
    nspiral: number of spirals, i.e. batch dimension number
    ntotal: total number of datapoints per spiral
    nsample: number of sampled datapoints for model fitting per spiral
    start: spiral starting theta value
    stop: spiral ending theta value
    noise_std: observation noise standard deviation
    to_tensor: convert to `torch.Tensor`
    a, b: parameters of the Archimedean spiral
    device: cpu/gpu device to copy tensor to

  Returns: 
    Tuple where first element is a tensor of size (nspiral, ntotal, 2)
    for true trajectories, second element is a tensor of size
    (nspiral, nsample, 2) for noisy observation, third element is
    a tensor of size (nspiral, ntotal) for all timestamps,
    and fourth element is a list of length `nspiral` for indices of initial
    timestamps
  """
  # sample a and b from Gaussian
  ts = np.linspace(start, stop, num=ntotal)  # (ntotal,)
  rs = a + b * ts  # (ntotal,)
  xs, ys = rs * np.cos(ts), rs * np.sin(ts)  # (ntotal,)
  orig_traj = np.stack((xs, ys), axis=1)  # (ntotal, 2)

  # sample starting timestamps
  samp_t0_ids = []
  samp_traj = []
  for _ in range(nspiral):
    t0_id = np.argmax(
        npr.multinomial(
            1, [1. / (ntotal - nsample - 1)] * (ntotal - nsample - 1)))
    samp_t0_ids.append(t0_id)
    traj = orig_traj[t0_id:t0_id + nsample, :].copy()
    traj += npr.randn(*traj.shape) * noise_std
    samp_traj.append(traj)
  samp_traj = np.stack(samp_traj, axis=0)

  if to_tensor:
    orig_traj = torch.from_numpy(orig_traj).float().to(device)
    samp_traj = torch.from_numpy(samp_traj).float().to(device)
    ts = torch.from_numpy(ts).float().to(device)

  return orig_traj, samp_traj, ts, samp_t0_ids


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


class RecognitionRNN(nn.Module):

  def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
    super(RecognitionRNN, self).__init__()

    self.nhidden = nhidden
    self.nbatch = nbatch
    self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
    self.h2o = nn.Linear(nhidden, latent_dim * 2)

  def forward(self, x, h):
    combined = torch.cat((x, h), dim=1)
    h = torch.tanh(self.i2h(combined))
    out = self.h2o(h)
    return out, h

  def initHidden(self):
    return torch.zeros(self.nbatch, self.nhidden)


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
  const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
  const = torch.log(const)
  return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
  v1 = torch.exp(lv1)
  v2 = torch.exp(lv2)
  lstd1 = lv1 / 2.
  lstd2 = lv2 / 2.

  kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
  return kl

if __name__ == '__main__':
  # constants
  latent_dim = 4
  nhidden = 20
  rnn_nhidden = 25
  obs_dim = 2
  noise_std = .1
  nspiral = 1000
  start = 0.
  stop = 4 * np.pi
  ntotal = 500
  nsample = 100

  device = torch.device('cuda:' + str(args.gpu)
                        if torch.cuda.is_available() else 'cpu')

  # generate toy data
  orig_traj, samp_traj, ts, samp_t0_ids = generate_spiral2d(
      nspiral=nspiral,
      start=start,
      stop=stop,
      noise_std=noise_std,
      to_tensor=True,
      device=device
  )

  # model
  func = LatentODEfunc(latent_dim, nhidden).to(device)
  rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nspiral).to(device)
  dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
  params = (list(func.parameters()) +
            list(dec.parameters()) + list(rec.parameters()))
  optimizer = optim.Adam(params, lr=args.lr)
  loss_meter = RunningAverageMeter()

  for itr in range(1, args.niters + 1):
    optimizer.zero_grad()
    # backward in time to infer q(z_0)
    h = rec.initHidden().to(device)
    for i in reversed(range(samp_traj.size(1))):
      obs = samp_traj[:, i, :]
      out, h = rec.forward(obs, h)
    qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
    epsilon = torch.randn(qz0_mean.size()).to(device)
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

    # forward in time and solve ode for reconstructions
    _min_t0_id, _max_t0_id = min(samp_t0_ids), max(samp_t0_ids)
    _z0 = z0[np.argmin(samp_t0_ids)]
    _ts = ts[_min_t0_id:_max_t0_id + nsample]
    _zs = odeint(func, _z0, _ts)

    pred_z = []
    for i in range(nspiral):
      t0_id = samp_t0_ids[i]
      t0_id_adjust = t0_id - _min_t0_id
      curr_zs = _zs[t0_id_adjust:t0_id_adjust + nsample]
      pred_z.append(curr_zs)
    pred_z = torch.stack(pred_z, dim=0)
    pred_x = dec(pred_z)

    # compute loss
    noise_std_ = torch.zeros(pred_x.size()).to(device)
    noise_std_ += noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(samp_traj, pred_x, noise_logvar).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
    loss = torch.mean(-logpx + analytic_kl, dim=0)
    loss.backward()
    optimizer.step()
    loss_meter.update(loss.item())
    print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))
  print("Training complete.")

  # sample latent traj from approx. posterior; sampling from the prior gives
  # bad performance
  with torch.no_grad():
    h = rec.initHidden().to(device)
    for t in reversed(range(samp_traj.size(1))):
      obs = samp_traj[:, t, :]
      out, h = rec.forward(obs, h)
    qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
    epsilon = torch.randn(qz0_mean.size()).to(device)
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

    t0_ = samp_ts[0, 0]
    t_pos = np.linspace(t0_, stop, num=3000)
    t_neg = np.linspace(start, t0_, num=3000)[::-1].copy()

    t_pos = torch.from_numpy(t_pos).float().to(device)
    t_neg = torch.from_numpy(t_neg).float().to(device)
    z_pos = odeint(func, z0, t_pos).permute(1, 0, 2)
    z_neg = odeint(func, z0, t_neg).permute(1, 0, 2)
    x_pos, x_neg = dec(z_pos), dec(z_neg)
    xs = torch.cat((torch.flip(x_neg, dims=[1]), x_pos), dim=1)

  xs = xs.cpu().numpy()
  orig_traj = orig_traj.cpu().numpy()
  samp_traj = samp_traj.cpu().numpy()

  # visualize
  plt.figure()
  plt.plot(orig_traj[:, 0], orig_traj[:, 1], 'g', label='true traj')
  plt.plot(xs[0, :, 0], xs[0, :, 1], 'r', label='learned traj')
  plt.scatter(
      samp_traj[0, :, 0], samp_traj[0, :, 1], s=3, label='sampled data')
  plt.legend()
  plt.show()
