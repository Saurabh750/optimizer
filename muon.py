from typing import Iterable

import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DTensor

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        params: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, params, muon_selector=None, lr=0.02, momentum=0.95, nesterov=True, ns_steps=6,
                 adamw_lr=3e-4, adamw_betas=[0.95, 0.95], adamw_eps=1e-8, adamw_wd=0):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr_ratio=adamw_lr/lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)

        if muon_selector is None:
            muon_selector = lambda name, param: (
                param.requires_grad and
                param.ndim >= 2 and                 # Check if scalar
                "embed" not in name.lower() and     # Check if embedding layer
                "tok" not in name.lower() and       # Check if token embeddings
                "head" not in name.lower() and      # Check if output head
                "bias" not in name.lower()          # Check if bias term
            )

        named_params = list(params)

        muon_params = [p for n, p in named_params if muon_selector(n, p)]
        adamw_params = [p for n, p in named_params if not muon_selector(n, p)]

        super().__init__([*muon_params, *adamw_params], defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        # we cant pickle booleans for saving, so we will use 1=True, 0=False
        def assign_muon(p):
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]['use_muon'] = 1
            else:
                self.state[p]['use_muon'] = 0

        if isinstance(muon_params[0], dict):
            for group in muon_params:
                for p in group['params']:
                    assign_muon(p)
        else:
            for p in muon_params:
                assign_muon(p)

        def assign_adamw(p):
            # Do not use Muon for parameters in adamw_params
            self.state[p]['use_muon'] = 0

        if len(adamw_params) and isinstance(adamw_params[0], dict):
            for group in adamw_params:
                for p in group['params']:
                    assign_adamw(p)
        else:
            for p in adamw_params:
                assign_adamw(p)

        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def to_dist(self, x, from_local=False, **meta):
        if from_local:
            return DTensor.from_local(
                x,
                device_mesh=meta["device_mesh"],
                placements=meta["placements"],
                shape=meta["shape"],
                stride=meta["stride"],
            )
        else:
            return distribute_tensor(x, device_mesh=meta["device_mesh"], placements=meta["placements"])


    def to_local(self, x, keep_sharded=False):
        if isinstance(x, DTensor):
            meta = dict(
                device_mesh=x.device_mesh,
                placements=x.placements,
                shape=x.shape,
                stride=x.stride(),
            )
            if keep_sharded:
                return x.to_local(), meta
            else:
                return x.full_tensor(), meta

        return x, None

    def setNorm(self, G, norm, p=None):
        if norm == 'spectral':
            return torch.linalg.norm(G, ord=2)
        elif norm == 'frobenius':
            return torch.linalg.norm(G, ord='fro')
        elif norm == 'nuclear':
            return torch.linalg.norm(G, ord='nuc')
        elif norm == 'spatial':
            if p is None: raise ValueError("p must be specified for spatial norm")
            return G.flatten(start_dim=2).norm(p=p, dim=2) 
        else:
            raise ValueError(f"Unknown norm type: {norm}. Supported norms are 'spectral', 'frobenius', and 'nuclear'.")

    def zeropower_via_newtonschulz5(self, G, steps=10, eps=1e-7):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16()
        X /= (X.norm() + eps) # ensure top singular value <= 1
        # X /= (self.setNorm(G, 'spectral') + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group['momentum']
            for i, p in enumerate(group['params']):
                if self.state[p]['use_muon'] == 1:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)

                    meta = None
                    if isinstance(g, DTensor):
                        g, meta = self.to_local(g, keep_sharded=False)
                    # gives NaNs when done with Dtensor, instead of throwing a typical op not supported error, quite sneaky
                    g = self.zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                    if meta is not None:
                        g = self.to_dist(g, **meta)
                    g *= max(1, g.size(0)/g.size(1))**0.5

                    g = g.view_as(p.data).type_as(p.data)
                    p.data.add_(g, alpha=-lr)
                else:
                    # these are all pointwise so we can stay in Dtensor
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['moment1'] = torch.zeros_like(g)
                        state['moment2'] = torch.zeros_like(g)
                    state['step'] += 1
                    step = state['step']
                    buf1 = state['moment1']
                    buf2 = state['moment2']
                    buf1.lerp_(g, 1-group['adamw_betas'][0])
                    buf2.lerp_(g.square(), 1-group['adamw_betas'][1])

                    g = buf1 / (group['adamw_eps'] + buf2.sqrt())

                    bias_correction1 = 1 - group['adamw_betas'][0]**step
                    bias_correction2 = 1 - group['adamw_betas'][1]**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.mul_(1 - lr * group['adamw_wd'])
                    p.data.add_(g, alpha=-lr/scale)

# Convert 1-D params to 2-D and dont use Adam optimizer
class MuonOneDimtoTwoDim(torch.optim.Optimizer):
    def __init__(self, params, muon_selector=None, lr=0.02, momentum=0.95, nesterov=True, ns_steps=6):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

        # Corrected muon_selector logic:
        # If a custom selector is not provided, we define one that includes
        # *all* parameters that require gradients.
        # This means all parameters will now attempt to be processed by Muon.
        # The logic for 1D to 2D conversion will then be handled in the step() method.
        if muon_selector is None:
            muon_selector = lambda name, param: param.requires_grad
        
        named_params = list(params)
        
        # Now, all parameters that require gradients (based on the new selector)
        # will be directed to the Muon processing path.
        muon_params = [p for n, p in named_params if muon_selector(n, p)]
        
        super().__init__(muon_params, defaults)

        def assign_param_type(p):
            self.state[p]['use_muon'] = 1 # Mark for Muon processing
            if p.ndim == 1:
                # This flag indicates that the parameter needs to be treated as
                # a diagonal matrix in the step() function.
                self.state[p]['is_diagonal_param'] = True
            else:
                self.state[p]['is_diagonal_param'] = False

        if len(muon_params) > 0 and isinstance(muon_params[0], dict):
            for group in muon_params:
                for p in group['params']:
                    assign_param_type(p)
        else:
            for p in muon_params:
                assign_param_type(p)

        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
            
    def to_dist(self, x, from_local=False, **meta):
        if from_local:
            return DTensor.from_local(
                x,
                device_mesh=meta["device_mesh"],
                placements=meta["placements"],
                shape=meta["shape"],
                stride=meta["stride"],
            )
        else:
            return distribute_tensor(x, device_mesh=meta["device_mesh"], placements=meta["placements"])


    def to_local(self, x, keep_sharded=False):
        if isinstance(x, DTensor):
            meta = dict(
                device_mesh=x.device_mesh,
                placements=x.placements,
                shape=x.shape,
                stride=x.stride(),
            )
            if keep_sharded:
                return x.to_local(), meta
            else:
                return x.full_tensor(), meta

        return x, None

    def setNorm(self, G, norm, p=None):
        if norm == 'spectral':
            return torch.linalg.norm(G, ord=2)
        elif norm == 'frobenius':
            return torch.linalg.norm(G, ord='fro')
        elif norm == 'nuclear':
            return torch.linalg.norm(G, ord='nuc')
        elif norm == 'spatial':
            if p is None: raise ValueError("p must be specified for spatial norm")
            return G.flatten(start_dim=2).norm(p=p, dim=2) 
        else:
            raise ValueError(f"Unknown norm type: {norm}. Supported norms are 'spectral', 'frobenius', and 'nuclear'.")

    def zeropower_via_newtonschulz5(self, G, steps=10, eps=1e-7):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16()
        X /= (X.norm() + eps) # ensure top singular value <= 1
        # X /= (self.setNorm(G, 'spectral') + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                # All parameters in param_groups should now be marked for Muon
                # due to __init__ modifications.
                
                g = p.grad
                if g is None:
                    continue

                is_diagonal_param = self.state[p].get('is_diagonal_param', False)
                
                # Standardize gradient to 2D for Muon processing
                if is_diagonal_param:
                    # For 1D parameters, create a diagonal matrix from the gradient
                    # Assert g is 1D, as per design
                    if g.ndim != 1:
                        raise ValueError(
                            f"Gradient for 1D parameter (flagged is_diagonal_param=True) "
                            f"is not 1D. Expected 1D, got {g.shape}."
                        )
                    g_2d = torch.diag(g)
                elif g.ndim > 2:
                    # For higher-dimensional parameters, flatten to 2D (as per original Muon)
                    g_2d = g.view(g.size(0), -1)
                else:
                    # For already 2D parameters
                    g_2d = g

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g_2d)
                buf = state['momentum_buffer']
                
                buf.mul_(momentum).add_(g_2d)

                if nesterov:
                    g_for_ns = g_2d.add(buf, alpha=momentum)
                else:
                    g_for_ns = buf

                # DTensor handling removed, assuming standard Tensors for this diagonal conversion
                # meta = None
                # if isinstance(g_for_ns, DTensor):
                #     g_for_ns, meta = self.to_local(g_for_ns, keep_sharded=False)

                g_orthogonalized = self.zeropower_via_newtonschulz5(g_for_ns, steps=ns_steps)

                # if meta is not None:
                #     g_orthogonalized = self.to_dist(g_orthogonalized, **meta)
                
                # Muon's specific scaling for rectangular matrices.
                # For diagonal matrices (where dim0 == dim1), this factor becomes 1.
                g_scaled = g_orthogonalized * max(1, g_orthogonalized.size(0)/g_orthogonalized.size(1))**0.5

                # Convert the processed 2D gradient back to the original parameter shape
                if is_diagonal_param:
                    # Extract diagonal from the 2D matrix back to 1D
                    update_tensor = torch.diag(g_scaled)
                    update_tensor = update_tensor.view_as(p.data).type_as(p.data)
                else:
                    # For original 2D or higher-dim parameters, reshape if necessary
                    update_tensor = g_scaled.view_as(p.data).type_as(p.data)
                
                # Apply the update
                p.data.add_(update_tensor, alpha=-lr)


class MuonSchattenp(torch.optim.Optimizer):
    """
    MuonSchattenp - Muon optimizer variant using schatten-p norm instead of Frobenius norm
    
    This variant replaces the Frobenius norm in Newton-Schulz iteration with schatten-p norm,
    which handles higher-dimensional tensors by computing norms across schatten dimensions.
    """
    def __init__(self, params, muon_selector=None, lr=0.02, momentum=0.95, nesterov=True, 
                 ns_steps=6, p=2):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, p=p)

        if muon_selector is None:
            muon_selector = lambda name, param: param.requires_grad
        
        named_params = list(params)
        muon_params = [p for n, p in named_params if muon_selector(n, p)]
        
        super().__init__(muon_params, defaults)

        def assign_param_type(p):
            self.state[p]['use_muon'] = 1
            if p.ndim == 1:
                self.state[p]['is_diagonal_param'] = True
            else:
                self.state[p]['is_diagonal_param'] = False

        if len(muon_params) > 0 and isinstance(muon_params[0], dict):
            for group in muon_params:
                for p in group['params']:
                    assign_param_type(p)
        else:
            for p in muon_params:
                assign_param_type(p)

        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def schatten_p_norm(self, X, p, eps=1e-7):
        """
        Compute Schatten-p norm of a 2D matrix.

        Args:
            X: Input tensor (2D matrix)
            p: The p value for the norm (can be float('inf') for spectral norm)
            eps: Small epsilon for numerical stability

        Returns:
            Schatten-p norm value
        """
        # Handle bfloat16 compatibility
        if X.dtype == torch.bfloat16:
            X_calc = X.float()
        else:
            X_calc = X

        if p == float('inf'):
            # Spectral norm = largest singular value
            return torch.linalg.norm(X_calc, ord=2) + eps
        else:
            # Compute singular values
            s = torch.linalg.svdvals(X_calc)
            return (torch.sum(s**p))**(1.0/p) + eps

    def zeropower_via_newtonschulz5_schatten(self, G, p, steps=10):
        """
        Newton-Schulz iteration using schatten-p norm instead of Frobenius norm.
        """
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        
        # Use schatten-p norm for normalization (handles bfloat16 internally)
        schatten_norm = self.schatten_p_norm(X, p)
        
        # Convert back to original dtype for division
        if X.dtype == torch.bfloat16:
            schatten_norm = schatten_norm.to(torch.bfloat16)
            
        X /= schatten_norm
        
        if G.size(0) > G.size(1):
            X = X.T
            
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
            
        if G.size(0) > G.size(1):
            X = X.T
            
        return X

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            p = group['p']  # schatten-p norm parameter

            for param in group['params']:
                g = param.grad
                if g is None:
                    continue

                is_diagonal_param = self.state[param].get('is_diagonal_param', False)
                
                # Standardize gradient to 2D for Muon processing
                if is_diagonal_param:
                    if g.ndim != 1:
                        raise ValueError(
                            f"Gradient for 1D parameter (flagged is_diagonal_param=True) "
                            f"is not 1D. Expected 1D, got {g.shape}."
                        )
                    g_2d = torch.diag(g)
                elif g.ndim > 2:
                    g_2d = g.view(g.size(0), -1)
                else:
                    g_2d = g

                state = self.state[param]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g_2d)
                buf = state['momentum_buffer']
                
                buf.mul_(momentum).add_(g_2d)

                if nesterov:
                    g_for_ns = g_2d.add(buf, alpha=momentum)
                else:
                    g_for_ns = buf

                # Use schatten-p norm in Newton-Schulz iteration
                g_orthogonalized = self.zeropower_via_newtonschulz5_schatten(
                    g_for_ns, p, steps=ns_steps
                )
                
                # Muon's specific scaling for rectangular matrices
                g_scaled = g_orthogonalized * max(1, g_orthogonalized.size(0)/g_orthogonalized.size(1))**0.5

                # Convert back to original parameter shape
                if is_diagonal_param:
                    update_tensor = torch.diag(g_scaled)
                    update_tensor = update_tensor.view_as(param.data).type_as(param.data)
                else:
                    update_tensor = g_scaled.view_as(param.data).type_as(param.data)
                
                # Apply the update
                param.data.add_(update_tensor, alpha=-lr)

        return loss

