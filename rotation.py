import torch
from typing import Optional

def r_matrix(theta: torch.Tensor) -> torch.Tensor:
    
    theta_tensor = theta.detach().clone() 
    theta2 = 2 * theta_tensor 
    c2 = torch.cos(torch.deg2rad(theta2)) 
    s2 = torch.sin(torch.deg2rad(theta2))
    B = theta.shape[0]
    
    R = torch.zeros(B, 3, 3, dtype=torch.complex64, device=theta.device)
    R[:, 0, 0] = 1.0
    R[:, 1, 1] = c2
    R[:, 1, 2] = s2
    R[:, 2, 1] = -s2
    R[:, 2, 2] = c2
    return R

def u_matrix(psi: torch.Tensor) -> torch.Tensor:
    
    psi_tensor = psi.detach().clone()
    psi2 = 2 * psi_tensor

    c2 = torch.cos(torch.deg2rad(psi2)) 
    s2 = torch.sin(torch.deg2rad(psi2))

    B = psi.shape[0]
    U = torch.zeros(B, 3, 3, dtype=torch.complex64, device=psi.device)
    U[:, 0, 0] = 1.0
    U[:, 1, 1] = c2
    U[:, 1, 2] = 1j * s2
    U[:, 2, 1] = 1j * s2
    U[:, 2, 2] = c2
    return U