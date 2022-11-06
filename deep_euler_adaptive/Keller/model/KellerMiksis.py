import math
import numpy as np
import torch
import torch.nn as nn

ro_L = 9.970639504998557e+02
P_inf = 1.0e+5
p_v = 3.166775638952003e+03
sigma = 0.071977583160056
R_E = 10.0/1.0e6
gam = 1.4
c_L = 1.497251785455527e+03
mu_L = 8.902125058209557e-04
theta = 0.0
P_A1 = 1e5
P_A2 = 0.0
f = 20e3 #20 kHz
f2 = 0.0

class Keller(nn.Module):
    '''The ODE function of the Keller-Miksis equation'''
    
    def __init__(self):
        super().__init__()
        
        C = np.zeros(13)
        twr = 1.0/(R_E*f)
        C[0] = (P_inf - p_v + 2*sigma/R_E)/ro_L*twr*twr
        C[1] = (1-3*gam)/(ro_L*c_L)*(P_inf - p_v + 2*sigma/R_E)*twr
        C[2] = (P_inf - p_v)/ro_L * twr*twr
        C[3] = 2*sigma/(ro_L*R_E) *twr*twr
        C[4] = 4*mu_L/(ro_L*R_E*R_E) / f
        C[5] = P_A1/ro_L * twr*twr
        C[6] = P_A2/ro_L *twr*twr
        C[7] = R_E * 2*math.pi*f * P_A1/(ro_L*c_L) * twr*twr
        C[8] = R_E * 2*math.pi*f * P_A2/(ro_L*c_L) * twr*twr
        C[9] = R_E*f / c_L
        C[10] = 3*gam
        C[11] = f2 / f
        C[12] = theta
        self.C = C
        
    def ode(self, t, x):
        '''For use with scipy.integrate.odeint'''
        C = self.C
        dxdt = np.ones(x.shape)
        rx0 = 1.0 / x[0];
        
        N = (C[0]+C[1]*x[1])*pow(rx0,C[10]) - C[2]*(1.0+C[9]*x[1]) -C[3]*rx0 -C[4]*x[1]*rx0 -\
        (1.5 - 0.5*C[9]*x[1])*x[1]*x[1] -\
        (C[5]*np.sin(2.0*math.pi*t) + C[6]*np.sin(2.0*math.pi*C[11]*t + C[12])) * (1.0+C[9]*x[1])-\
        x[0] * (C[7]*np.cos(2.0*math.pi*t) + C[8]*np.cos(2.0*math.pi*C[11]*t+C[12]) );
        
        D = x[0] - C[9]*x[0]*x[1] + C[4]*C[9];
                
        dxdt[0] = x[1]
        dxdt[1] = N / D
        return dxdt
    
    def forward(self, t, x):
        '''Module formulation with tensors for use with torchdiffeq odeint'''
        C = self.C
        dxdt = torch.ones(1,2)
        rx0 = 1.0 / x[0,0];
        
        N = (C[0]+C[1]*x[0,1])*pow(rx0,C[10]) - C[2]*(1.0+C[9]*x[0,1]) -C[3]*rx0 -C[4]*x[0,1]*rx0 -\
        (1.5 - 0.5*C[9]*x[0,1])*x[0,1]*x[0,1] -\
        (C[5]*torch.sin(2.0*math.pi*t) + C[6]*torch.sin(2.0*math.pi*C[11]*t + C[12])) * (1.0+C[9]*x[0,1])-\
        x[0,0] * (C[7]*torch.cos(2.0*math.pi*t) + C[8]*torch.cos(2.0*math.pi*C[11]*t+C[12]) );
        
        D = x[0,0] - C[9]*x[0,0]*x[0,1] + C[4]*C[9];
                
        dxdt[0,0] = x[0,1]
        dxdt[0,1] = N / D
        return dxdt