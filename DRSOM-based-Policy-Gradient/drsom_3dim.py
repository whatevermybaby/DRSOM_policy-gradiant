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
# from shanshu.MBPG.Algorithms.MBPG import compute_weights
import torch
from torch.nn.utils import parameters_to_vector

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DRSOM_VERBOSE = int(os.environ.get('DRSOM_VERBOSE', 1))


def _norm(alpha, tr):
  # print(alpha)
  # print(tr)
  return (tr @ alpha).dot(alpha).sqrt()


def _compute_root(Q, c, gamma, tr=torch.eye(3)):
  # lsolve = torch.linalg.solve
  lsolve = np.linalg.solve
  
  # D, V = torch.linalg.eigh(Q)
  D, V = np.linalg.eigh(Q)
  print("特征值分别是")
  print(D)
  lmin, lmedium,lmax = D
  lb = max(0, -lmin.item())
  lmax = lmax.item() if lmax > lb else lb + 1e4
  _lmb_this = gamma * lmax + max(1 - gamma, 0) * lb
  it = 0
  try:
    alpha = lsolve(Q + tr * _lmb_this, -c)
    print("算出来的就是")
    print(alpha)
  except np.linalg.LinAlgError as e:
    print(e)
    print(Q, tr, _lmb_this, -c)
    # todo, can we do better
    alpha = lsolve(Q + tr * (_lmb_this + 1e-4), -c)

  alpha_tensor = torch.Tensor(alpha.copy())

  # norm = _norm(alpha, tr)
  norm = _norm(alpha_tensor, tr)
  
  # return it, _lmb_this, alpha, norm, True
  return it, _lmb_this, alpha_tensor, norm, True


class DRSOMF(torch.optim.Optimizer):
  
  def __init__(
      self,
      params,
      max_iter=15,
      option_tr='a',
      gamma=1e-6,
      beta1=5e1,
      beta2=3e1,
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
      self.state[p]['oldg']=torch.zeros_like(p.data, requires_grad=True)
    
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
    self.Q: Optional[torch.Tensor] = torch.zeros((3, 3), requires_grad=False)
    self.c: Optional[torch.Tensor] = torch.zeros(3, requires_grad=False)
    self.G: Optional[torch.Tensor] = torch.zeros((3, 3), requires_grad=False)
    
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
    # self.eta = 0.08
    self.eta = 0.01
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
  
  def _apply_step(self, flat_p, flat_d,flat_g):
    with torch.no_grad():
      offset = 0
      for p in self._params:
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        p.copy_(flat_p[offset:offset + numel].view_as(p))
        self.state[p]['momentum'].copy_(flat_d[offset:offset + numel].view_as(p))
        self.state[p]['oldg'].copy_(flat_g[offset:offset + numel].view_as(p))
        offset += numel
      assert offset == self._numel()
  
  def _directional_evaluate(self, closure, flat_p, flat_d,flat_g):
    self._apply_step(flat_p, flat_d,flat_g)
    # evaluation
    loss = float(closure(backward=False))
    return loss
  
  def _numel(self):
    if self._numel_cache is None:
      self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
    return self._numel_cache
  
  def _gather_flat_grad(self, _valid_params, target='self'):
    if target == 'grad':
      # flat = torch.concat([p.grad.reshape(-1) for p in _valid_params])
      flat = torch.cat([p.grad.reshape(-1) for p in _valid_params])
    elif target == 'momt':
      # flat = torch.concat([self.state[p]['momentum'].reshape(-1) for p in _valid_params])
      flat = torch.cat([self.state[p]['momentum'].reshape(-1) for p in _valid_params])
    elif target == 'oldg':
      flat = torch.cat([self.state[p]['oldg'].reshape(-1) for p in _valid_params])
    else:
      # flat = torch.concat([p.reshape(-1) for p in _valid_params])
      flat = torch.cat([p.reshape(-1) for p in _valid_params])
    
    return flat
  
  @torch.no_grad()
  def solve_alpha(self, Q, c, tr=torch.eye(3)):
    # initialization
    # if self.iter == 0 or self.Q[1, 1] < 1e-4:
    if self.iter == 0:
      lmd = 0.0
      alpha = torch.tensor([-c[0] / Q[0, 0] / (1 + self.gamma), 0,0])  if Q[0, 0] > 0 else torch.tensor([1e-4, 0,0])
      norm = _norm(alpha, tr)
      if norm > self.delta_max:
        print('yes')
        alpha = alpha / alpha.norm() * self.delta_max
    else:
      # apply root-finding
      it, lmd, alpha, norm, active = _compute_root(Q, c, self.gamma, tr)
      # alpha=alpha*10
    
    if DRSOM_VERBOSE:
    # if 1:
      self.logline = {
        '𝜆': '{:+.2e}'.format(lmd),
        'Q/c/G': np.round(np.vstack([self.Q, self.c, self.G]), 3),
        'a': np.round(alpha.tolist(), 5).reshape((3, 1)),
        'ghg': '{:+.2e}'.format(Q[0, 0]),
        'ghg-': '{:+.2e}'.format(self.ghg),
      }
    return alpha, norm
  
  def compute_step(self, option_tr='p'):
    # compute alpha
    if option_tr == 'a':
      self.alpha, self.alpha_norm = self.solve_alpha(
        self.Q, self.c,
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
  
  def update_trust_region(self, flat_p, flat_g, flat_d,flat_v,g_norm, d_norm,v_norm):
    with torch.enable_grad():
      __unused = flat_p
      # size = flat_g.size()[0]//2
      # gsort = (flat_g * flat_g).sort()
      # gindx = gsort[1][-size:]
      # flat_g=flat_g*1000/flat_g.dot(flat_g).sqrt()
      # flat_d=flat_d/_norm(flat_d,tr=torch.eye(2))
      gg = flat_g.dot(flat_g)
      gd = flat_d.dot(flat_g)
      dd = flat_d.dot(flat_d)
      gv = flat_g.dot(flat_v)
      dv = flat_d.dot(flat_v)
      vv = flat_v.dot(flat_v)
      
      if self.iter % self.freq == 0:
        # @note:
        #   compute Hv:
        #   by analytic gg gd...
        Hg = self._gather_flat_grad(torch.autograd.grad(
          # (flat_g[gindx] ** 2).sum()/ 2, self._params,
          (flat_g ** 2).sum() / 2, self._params,
          # create_graph=True,
          retain_graph=True
        ), target='self')
        Hd = self._gather_flat_grad(torch.autograd.grad(
          gd, self._params,
          # create_graph=True,
          retain_graph=True,
        ), target='self')
        Hv = self._gather_flat_grad(torch.autograd.grad(
          gv, self._params,
          # create_graph=True,
          retain_graph=True,
        ), target='self')
        # @note:
        # kept for information.
        # An alternative approach.
        #   for Hessian-vector via Jacobian (not good : ( )
        # Hg = self._gather_flat_grad(torch.autograd.grad(
        #   flat_g, self._params, flat_g,
        #   create_graph=True
        # ), target='self')
        # Hd = self._gather_flat_grad(torch.autograd.grad(
        #   flat_g, self._params, flat_d,
        #   create_graph=True
        # ), target='self')
        
        gHg = flat_g.dot(Hg)/g_norm**2
        gHd = flat_g.dot(Hd)/ g_norm / d_norm
        dHd = flat_d.dot(Hd)/ d_norm**2
        gHv =flat_g.dot(Hv)/g_norm/v_norm
        dHv=flat_d.dot(Hv)/d_norm/v_norm
        vHv=flat_v.dot(Hv)/v_norm**2
        Q = torch.tensor([[gHg, -gHd,-gHv], [-gHd, dHd,dHv],[-gHv,dHv,vHv]], requires_grad=False)
        # Q = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=False)

      else:
        # if set freq = 1
        #   this never happens
        Q = torch.tensor([[gg, 0.0,0.0], [0.0, dd,0.0],[0.0,0.0,vv]], requires_grad=False)
        # Q = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=False)

      
      # c = torch.tensor([-gg, gd], requires_grad=False)
      c = torch.tensor([-gg/g_norm, gd/d_norm,gv/v_norm], requires_grad=False)
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
      G = torch.tensor([[gg/g_norm**2, -gd/g_norm/d_norm,-gv/g_norm/d_norm], [-gd/g_norm/d_norm, dd/d_norm**2,dv/d_norm/v_norm],[-gv/g_norm/v_norm,dv/d_norm/v_norm,vv/d_norm**2]])
      self.Ga.append(G)
      self.G = sum(_G * b[k] for k, _G in enumerate(self.Ga))
  
  def compute_weights(o_lh, c_lh, th=1.8):
    #mini_bs = o_lh.size()
    lth = th - 1
    #3lth = max(lth, 0)
    o_lh_sum = o_lh.sum(dim=1)
    c_lh_sum = c_lh.sum(dim=1)
    weigt = o_lh_sum - c_lh_sum
    weigt = torch.exp(weigt)
    weigt[weigt>=th] = th
    #weigt[weigt<lth] = lth
    print(weigt.max())
    return weigt.unsqueeze(1).expand_as(c_lh)

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
    # if 1:
      torch.autograd.set_detect_anomaly(True)
    n_iter = 0
    
    
    loss=closure()
    # IS_w=compute_weights(old_likelihood,cl_hood,th=1.8)
    flat_g = parameters_to_vector([p.grad for p in self._params])
    flat_p = parameters_to_vector(self._params)
    flat_d = self._gather_flat_grad([self.state[p]['momentum'] for p in self._params])
    flat_v=flat_g-self._gather_flat_grad([self.state[p]['oldg'] for p in self._params])
    # print(flat_g)
    # print("接下来是v")
    # print(flat_v)
    # copy of it at last step
    p_copy = self._clone_param()

    g_norm = np.linalg.norm(flat_g.detach().numpy())
    d_norm = np.linalg.norm(flat_d.detach().numpy())
    v_norm = np.linalg.norm(flat_d.detach().numpy())
    d_norm = 1 if d_norm == 0 else d_norm
    v_norm = 1 if v_norm == 0 else v_norm

    self.update_trust_region(flat_p, flat_g, flat_d,flat_v,g_norm,d_norm,v_norm)
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
        # if 1:
          print(self.logline)
        continue
      alpha1, alpha2, alpha3= self.alpha
      
      # build direction
      # flat_new_g=flat_g/g_norm
      flat_new_d = torch.zeros_like(flat_d, requires_grad=False)
      flat_new_d.add_(flat_g/g_norm, alpha=-alpha1).add_(flat_d/d_norm, alpha=alpha2).add_(flat_v/v_norm,alpha=alpha3)
      flat_new_p = torch.zeros_like(flat_p, requires_grad=False).copy_(flat_p).add_(flat_new_d)
      
      # accept or not？
      loss_est = self._directional_evaluate(closure, flat_new_p, flat_new_d,flat_g)
      loss_dec = loss - loss_est
      rho = loss_dec / trs_est
      
      # update the trust-region radius (implicitly by gamma/lambda)
      lmb_dec = 0
      gamma_old = self.gamma
      if rho <= self.zeta1:
        self.gamma = max(self.gamma * self.beta1, 1e-4)
      else:
        if rho >= self.zeta2:
          lmb_dec = 1
          ## 调得激进一点
          self.gamma = max(self.gammalb, min(self.gamma / self.beta2, np.log(self.gamma)))
      
      acc_step = rho > self.eta
      if DRSOM_VERBOSE:
      # if 1:
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
