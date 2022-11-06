import numpy as np
from scipy.integrate import ode
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step, norm, warn_extraneous, validate_first_step)
from scipy.integrate._ivp.base import OdeSolver, DenseOutput

from utils.scalers import StandardScaler

import torch.jit

#torch.set_default_dtype(torch.float64)


SAFETY = 0.9 # Multiply steps computed from asymptotic behaviour of errors by this.

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

class DeepEuler(OdeSolver):
    MODE_GENERAL = 2
    MODE_AUTONOMOUS = 0
    MODE_ABSOLUTE_TIMES = 1
    
    def __init__(self, fun, t0, y0, t_bound, h, traced_model_path, scaler_path="", mode=MODE_GENERAL, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized=False, support_complex=False)
        
        self.h = h
        self.mode = mode

        self.model = torch.jit.load(traced_model_path)
        self.model.eval()
        if scaler_path:
            f = open(scaler_path,'r')
            self.out_scaler = StandardScaler(f)
            self.in_scaler = StandardScaler(f)
            f.close()
        else:
            self.out_scaler = False
        
    def _step_impl(self):
        t = self.t
        y = self.y
        h = self.h
        
        dydt = self.fun(t, y)
        
        if self.mode==DeepEuler.MODE_AUTONOMOUS:
            y_in = np.concatenate(([h], y))
        elif self.mode==DeepEuler.MODE_ABSOLUTE_TIMES:
            y_in = np.concatenate(([t+h], [t], y))
        else:
            y_in = np.concatenate(([t], [h], y))
            
        if self.out_scaler:
            model_out = self.out_scaler.inverse_transform(self.model(torch.tensor(self.in_scaler.transform(y_in))).detach().numpy())
        else:
            model_out = self.model(torch.tensor(y_in)).detach().numpy()
        
        y_new = y + h * dydt + h * h * model_out
        t_new = t + h
        
        self.y_old = y

        self.t = t_new
        self.y = y_new
        
        return (True, None)
        
    def _dense_output_impl(self):
        
        pass
       


       
class AdaptiveDeepEuler(OdeSolver):
    MODE_GENERAL = 2
    MODE_AUTONOMOUS = 0
    MODE_ABSOLUTE_TIMES = 1
    
    def __init__(self, fun, t0, y0, t_bound, traced_model_path, scaler_path="", max_step=np.inf, rtol=1e-3, atol=1e-4, first_step=1e-4, mode=MODE_GENERAL, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized=False, support_complex=False)
        
        self.mode = mode
        self.direction = 1

        self.model = torch.jit.load(traced_model_path)
        self.model.eval()
        if scaler_path:
            f = open(scaler_path,'r')
            self.out_scaler = StandardScaler(f)
            self.in_scaler = StandardScaler(f)
            f.close()
        else:
            self.out_scaler = False
            
        self.y_old = None
        self.h_previous = None
        
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)

        self.h_abs = validate_first_step(first_step, t0, t_bound)
        
        self.error_exponent = -1 / 2
        
    def _step_impl(self):
        t = self.t
        y = self.y
        
        dydt = self.fun(t, y) #calculated only once
        
        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, err = self._one_step(t, y, dydt, h)
            
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = norm(np.abs(err) / scale)
            
            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True
        
        self.t_old = t
        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        return True, None
    
    
    def _one_step(self, t, y, dydt, h):
        ''' One step with error estimation'''
        
        if self.mode==DeepEuler.MODE_AUTONOMOUS:
            y_in = np.concatenate(([h], y))
        elif self.mode==DeepEuler.MODE_ABSOLUTE_TIMES:
            y_in = np.concatenate(([t+h], [t], y))
        else: #General
            y_in = np.concatenate(([t], [h], y))
            
        if self.out_scaler:
            model_out = self.out_scaler.inverse_transform(self.model(torch.tensor(self.in_scaler.transform(y_in))).detach().numpy())
        else:
            model_out = self.model(torch.tensor(y_in)).detach().numpy()
        
        err = h * h * model_out
        y_new = y + h * dydt + err
        return y_new, err
        
        
    def _dense_output_impl(self):
        
        pass
    