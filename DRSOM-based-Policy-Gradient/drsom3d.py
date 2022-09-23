"""
radius free (dimension-reduced trust-region method) DRSOM
@author: Chuwen Zhang<chuwzhang@gmail.com>, Yinyu Ye<yinyu-ye@stanford.edu>
@note:
  This is a vanilla implementation of (Mini-batch, Radius-Free) DRSOM.
  
"""
import os
from collections import deque
from functools import reduce
from pprint import pprint
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DRSOM_VERBOSE = int(os.environ.get('DRSOM_VERBOSE', 0))


def _norm(alpha, tr):
  return (tr @ alpha).dot(alpha).sqrt()


def _compute_root(Q, c, gamma, tr=torch.eye(2)):
  lsolve = np.linalg.solve
  
  D, V = np.linalg.eigh(Q)
  
  lmin, lmax = min(D), max(D)
  print("smallest eigenvalue",lmin)
  lb = max(0, -lmin.item())
  lmax = lmax.item() if lmax > lb else lb + 1e4
  _lmb_this = gamma * lmax + max(1 - gamma, 0) * lb
  it = 0
  try:
    alpha = lsolve(Q + tr * _lmb_this, -c)
  except np.linalg.LinAlgError as e:
    print(e)
    print(Q, tr, _lmb_this, -c)
    # todo, can we do better
    alpha = lsolve(Q + tr * (_lmb_this + 1e-4), -c)
  alpha_tensor = torch.Tensor(alpha.copy())
  
  norm = _norm(alpha_tensor, tr)
  
  return it, _lmb_this, alpha_tensor, norm, True


class DRSOMF(torch.optim.Optimizer):
  
  def __init__(
      self,
      params,
      max_iter=15,
      option_tr='p',
      gamma=1e-8,
      beta1=2,
      beta2=2,
      hessian_window=1,
      thetas=(0.99, 0.999), eps=1e-8
  ):
    """
    The DRSOMF:
      Implementation of (Mini-batch) DRSOM (Dimension-Reduced Second-Order Method) in F (Radius-Free) style
    Args:
      params: model params
      max_iter: # of iteration for trust-region adjustment
      option_tr: option of trust-region, I or G?
               - if 'a'; G = eye(2)
               - if 'p'; G = [-g d]'[-g d]
      gamma: lower bound for gamma
      beta1: gamma + multiplier
      beta2: gamma - multiplier
      hessian_window: window size to keep last k hessian information
      thetas: weight decay params (like betas for Adam)
      eps: ...
    """
    
    defaults = dict(betas=thetas, eps=eps)
    super(DRSOMF, self).__init__(params, defaults)
    
    self._params = self.get_params()
    for p in self._params:
      # keep momentum
      self.state[p]['momentum'] = torch.zeros_like(p.data, requires_grad=True)
      self.state[p]['oldg'] = torch.zeros_like(p.data, requires_grad=True)
    
    #
    self._numel_cache = None
    ##########################
    # DRSOM only params
    ##########################
    # frequency to update Hv (Hessian-vector product)
    self.freq = 1
    self._max_iter_adj = max_iter
    self.option_tr = option_tr
    
    ##########################
    # global averages & keepers
    ##########################
    self.Q: Optional[torch.Tensor] = torch.zeros((2, 2), requires_grad=False)
    self.c: Optional[torch.Tensor] = torch.zeros(2, requires_grad=False)
    self.G: Optional[torch.Tensor] = torch.zeros((2, 2), requires_grad=False)
    
    ##########################
    # scalar attrs
    ##########################
    # total number of runs acc. all steps
    self.iter = 0
    self.alpha: Optional[torch.TensorType] = None
    self.alpha_norm = 0.0
    # gamma & lower bound on gamma
    self.gamma = gamma
    self.gammalb = 1e-12
    # gamma increasing rules
    self.beta1 = beta1
    self.beta2 = beta2
    # maximum step size
    self.delta_max = 1e1
    ##########################
    # step acc rules
    ##########################
    self.eta = 0.08
    self.zeta1 = 0.25
    self.zeta2 = 0.75
    ##########################
    # weight decay of the past
    ##########################
    self.hessian_window = hessian_window
    self.Qa = deque(maxlen=hessian_window)
    self.ca = deque(maxlen=hessian_window)
    self.Ga = deque(maxlen=hessian_window)
    self.thetas = thetas
    # other indicators
    self.ghg = 0.0
    # structured log line
    self.logline = None
  
  def get_params(self):
    """
    gets all parameters in all param_groups with gradients requirements
    """
    return [p for group in self.param_groups for p in group['params'] if p.requires_grad]
  
  def _clone_param(self):
    return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
  
  @torch.no_grad()
  def _set_param(self, params_data):
    
    for p, pdata in zip(self._params, params_data):
      p.copy_(pdata)
  
  def _bool_grad_vanish(self, p):
    return p.grad is None or torch.linalg.norm(p.grad) < 1e-8
  
  @torch.no_grad()
  def _clear_momentum(self):
    # only has globally state
    for p in self._params:
      self.state[p]['momentum'].zero_()
      self.state[p]['oldg'].zero_()
  
  def _apply_step(self, flat_p, flat_d, flat_oldg):
    with torch.no_grad():
      offset = 0
      for p in self._params:
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        p.copy_(flat_p[offset:offset + numel].view_as(p))
        self.state[p]['momentum'].copy_(flat_d[offset:offset + numel].view_as(p))
        self.state[p]['oldg'].copy_(flat_oldg[offset:offset + numel].view_as(p))
        offset += numel
      assert offset == self._numel()
  
  def _directional_evaluate(self, closure, flat_p, flat_d, flat_oldg):
    self._apply_step(flat_p, flat_d, flat_oldg)
    # evaluation
    loss = float(closure(backward=False))
    return loss
  
  def _numel(self):
    if self._numel_cache is None:
      self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
    return self._numel_cache
  
  def _gather_flat_grad(self, _valid_params, target='self'):
    if target == 'grad':
      flat = torch.cat([p.grad.reshape(-1) for p in _valid_params])
    elif target == 'momt':
      flat = torch.cat([self.state[p]['momentum'].reshape(-1) for p in _valid_params])
    elif target == 'oldg':
      flat = torch.cat([self.state[p]['oldg'].reshape(-1) for p in _valid_params])
    else:
      flat = torch.cat([p.reshape(-1) for p in _valid_params])
    
    return flat
  
  @torch.no_grad()
  def solve_alpha(self, Q, c, tr):
    # initialization
    dim = c.size()[0]
    if self.iter == 0 :# orself.Q[1, 1] < 1e-4:
      lmd = 0.0
      alpha = torch.zeros_like(c)
      alpha[0] = -c[0] / Q[0, 0] / (1 + self.gamma) if Q[0, 0] > 0 else 1e-4
      norm = _norm(alpha, tr)
      if norm > self.delta_max:
        alpha = alpha / alpha.norm() * self.delta_max
    else:
      # apply root-finding
      it, lmd, alpha, norm, active = _compute_root(Q, c, self.gamma, tr)
    
    if DRSOM_VERBOSE:
      self.logline = {
        '𝜆': '{:+.2e}'.format(lmd),
        'Q/c/G': np.round(np.vstack([self.Q, self.c, self.G]), 3),
        'a': np.round(alpha.tolist(), 3).reshape((dim, 1)),
        'ghg': '{:+.2e}'.format(Q[0, 0]),
        'ghg-': '{:+.2e}'.format(self.ghg),
      }
    return alpha, norm
  
  def compute_step(self, option_tr='p'):
    # compute alpha
    if option_tr == 'a':
      self.alpha, self.alpha_norm = self.solve_alpha(
        self.Q, self.c, tr=torch.eye(len(self.c))
      )
    elif option_tr == 'p':
      self.alpha, self.alpha_norm = self.solve_alpha(
        self.Q, self.c, tr=self.G
      )
    else:
      raise ValueError(f"unknown option for trust-region option: {option_tr}")
    
    ####################################
    # compute estimate decrease
    ####################################
    trs_est = - 1 / 2 * (self.Q @ self.alpha).dot(self.alpha) - self.c.dot(self.alpha)
    
    return trs_est
  
  def hv(self, gv, flag):
    mul = 0.5 if flag == 1 else 1
    hv = self._gather_flat_grad(torch.autograd.grad(
      gv * mul, self._params,
      # create_graph=True,
      retain_graph=True
    ), target='self')
    
    return hv
  
  def update_trust_region(self, flat_p, flat_g, directions, ad_flags):
    with torch.enable_grad():
      __unused = flat_p
      # size = flat_g.size()[0]//2
      # gsort = (flat_g * flat_g).sort()
      # gindx = gsort[1][-size:]
      dim = len(directions)
      # construct G (the inner products)
      G = torch.zeros((dim, dim), requires_grad=False, device='cpu')
      for i, v in enumerate(directions):
        for j in range(i, dim):
          u = directions[j]
          # compute G[i,j]
          G[i, j] = G[j, i] = v.dot(u).detach().cpu()
      
      # compute Hv for v in directions;
      #   assume directions[0] = g/|g|
      if self.iter % self.freq == 0:
        # @note:
        #   compute Hv:
        #   by analytic gv
        # if v := g, then should be scaled by 0.5
        Q = torch.zeros((dim, dim), requires_grad=False, device='cpu')
        Hv = [self.hv(flat_g.dot(v), ad_flags[i]) for i, v in enumerate(directions)]
        for i, v in enumerate(directions):
          for j in range(i, dim):
            Q[i, j] = Q[j, i] = v.dot(Hv[j]).detach().cpu()
      else:
        # if set freq = 1
        #   this never happens
        # Q = torch.tensor([[G, 0.0], [0.0, dd]], requires_grad=False)
        raise ValueError("not handled yet")
      # free memory
      del Hv
      c = torch.tensor([flat_g.dot(v) for v in directions], requires_grad=False)
      self.ghg = (Q[0, 0] + self.ghg * self.iter) / (self.iter + 1)
      self.Qa.appendleft(Q)
      self.ca.appendleft(c)
      
      # compute Q/c/G
      _total = len(self.Qa)
      beta1, beta2 = self.thetas
      b = torch.tensor([beta1 ** (k + 1) for k in range(len((self.Qa)))])
      b = b / b.sum()
      self.Q = sum(_Q * b[k] for k, _Q in enumerate(self.Qa))
      self.c = sum(_c * b[k] for k, _c in enumerate(self.ca))
      # use generalized a'Ga <= delta
      self.Ga.append(G)
      self.G = sum(_G * b[k] for k, _G in enumerate(self.Ga))
  
  def normalize(self, v):
    v_norm = torch.linalg.norm(v)
    v_norm = 1 if v_norm == 0 else v_norm
    return v / v_norm
  
  def step(self, closure=None):
    """
    Performs a single optimization step.
    Arguments:
        closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
    """
    
    if closure is None:
      raise ValueError("must provide a closure for RSOM")
    closure = torch.enable_grad()(closure)
    if DRSOM_VERBOSE:
      torch.autograd.set_detect_anomaly(True)
    n_iter = 0
    
    loss = closure()
    flat_p = parameters_to_vector(self._params)
    # copy of it at last step
    p_copy = self._clone_param()
    flat_g = parameters_to_vector([p.grad for p in self._params])
    
    # form directions
    flat_d = self._gather_flat_grad([self.state[p]['momentum'] for p in self._params])
    flat_oldg = self._gather_flat_grad([self.state[p]['oldg'] for p in self._params])
    
    # directions = [
    # self.normalize(- flat_g),
    # # self.normalize(flat_d)
    # # ]
    # # ad_flags = (1, 0)
    directions = [
      self.normalize(- flat_g),
      self.normalize(flat_d),
      # self.normalize(flat_g - flat_oldg)
    ]
    ad_flags = (1, 0, 0)
    
    self.update_trust_region(flat_p, flat_g, directions, ad_flags)
    # accept or not?
    acc_step = False
    # adjust lambda: (and thus trust region radius)
    iter_adj = 1
    while iter_adj < self._max_iter_adj:
      
      # solve alpha
      trs_est = self.compute_step(option_tr=self.option_tr)
      if trs_est < 0:
        self.gamma = max(self.gamma * self.beta1, 1e-4)
        if DRSOM_VERBOSE:
          pprint(self.logline)
        continue
      alpha = self.alpha
      
      # build direction
      flat_new_d = torch.zeros_like(flat_d, requires_grad=False)
      for aa, dd in zip(alpha, directions):
        flat_new_d.add_(dd, alpha=aa)
      
      # new trial points
      flat_new_p = torch.zeros_like(flat_p, requires_grad=False).copy_(flat_p).add_(flat_new_d)
      
      # accept or not？
      loss_est = self._directional_evaluate(closure, flat_new_p, flat_new_d, flat_g)
      loss_dec = loss - loss_est
      rho = loss_dec / trs_est
      
      # update the trust-region radius (implicitly by gamma/lambda)
      lmb_dec = 0
      gamma_old = self.gamma
      if rho <= self.zeta1:
        self.gamma = max(self.gamma * self.beta1, 1e-4)
        print("yes")
        print(self.gamma)
      else:
        if rho >= self.zeta2:
          lmb_dec = 1
          self.gamma = max(self.gammalb, min(self.gamma / self.beta2, np.log(self.gamma)))
      print('算出来的东西',rho)
      print('半径参数是',self.gamma)
      
      acc_step = rho > self.eta
      if DRSOM_VERBOSE:
        self.logline['dQ'] = '{:+.2e}'.format(trs_est.item())
        self.logline['df'] = '{:+.2e}'.format(loss_dec.item())
        self.logline['rho'] = '{:+.2e}'.format(rho.item())
        self.logline['acc'] = int(acc_step.item())
        self.logline['acc-𝜆'] = lmb_dec
        self.logline['𝛄'] = '{:+.2e}'.format(self.gamma)
        self.logline['𝛄-'] = '{:+.2e}'.format(gamma_old)
        self.logline['f'] = '{:+.2e}'.format(loss.item())
        self.logline['k'] = '{:+6d}'.format(self.iter)
        self.logline['k0'] = iter_adj
        print(
          pd.DataFrame(
            data=[list(self.logline.values())], columns=self.logline.keys(), dtype=str
          ).to_markdown(
            tablefmt="grid"
          )
        )
      if not acc_step:
        # set back to old ~ trial step failed
        self._set_param(p_copy)
      
      else:
        break
      
      iter_adj += 1
    
    self.iter += 1
    n_iter += 1
    
    if not acc_step:
      # if this step is not acc. (after max # of iteration for adjustment)
      # consider restart the optimizer by clearing the momentum,
      #   just like a nonlinear conjugate gradient method.
      self._clear_momentum()
      self.gamma = self.gammalb
    
    return loss
