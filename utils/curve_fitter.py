import numpy as np  
from scipy.interpolate import make_lsq_spline, BSpline  


class BSplineFitter:
    """
    Performs Least-Squares B-spline fitting for fixed-length multi-dimensional time series.
    
    Parameters
    ----------
    T : int
        Number of timesteps (sequence length). Subsequent calls to `fit()` must use this length.
    k : int, default 3
        B-spline degree (degree = k).
    n_ctrl : int, default 8
        Number of control points. Must satisfy n_ctrl >= k + 1.
    """
    def __init__(self, T: int, k: int = 3, n_ctrl: int = 8):
        if n_ctrl < k + 1:
            raise ValueError("n_ctrl must be at least k + 1")

        self.T = T
        self.k = k
        self.n_ctrl = n_ctrl

        self.x = np.arange(T, dtype=float)

        # ----------- Knot vector t (open uniform / clamped) -----------
        # Number of internal knots = n_ctrl - k - 1
        t_internal = np.linspace(self.x[0], self.x[-1], n_ctrl - k + 1)
        # Repeat endpoints k times to ensure the B-spline is clamped (multiplicity k+1)
        self.t = np.concatenate(([self.x[0]] * k, t_internal,
                                 [self.x[-1]] * k)).astype(float)

        # ----------- Greville Abscissae -----------
        self.ctrl_x = np.array([self.t[j + 1: j + k + 1].mean()
                                for j in range(n_ctrl)],
                               dtype=np.float32)
        
        self._last_spline: BSpline | None = None

    def fit(self, y: np.ndarray):
        """
        Parameters
        ----------
        y : ndarray, shape (T, D)
            Data to be fitted. T must match the initialized length.
        
        Returns
        -------
        ctrl_y : ndarray, shape (n_ctrl, D), dtype float32
            Control point coefficients in the least-squares sense.
        """
        y = np.asarray(y, dtype=float)
        if y.shape[0] != self.T:
            raise ValueError(f"y must have length {self.T} on axis 0")

        spline = make_lsq_spline(self.x, y, self.t, self.k, axis=0)

        self._last_spline = spline

        ctrl_y = spline.c.astype(np.float32)    # shape (n_ctrl, D)
        return ctrl_y

    def rebuild(self,
                ctrl_y: np.ndarray | None = None,
                dtype=np.float32):
        """
        Reconstruct the full curve based on control point coefficients.

        If `ctrl_y` is None, use the spline from the most recent `fit()` call.

        Parameters
        ----------
        ctrl_y : ndarray or None
            Shape must be (n_ctrl,) or (n_ctrl, D). If None, 
            reconstructs using `_last_spline`.
        dtype : numpy dtype, default np.float32
            Data type of the returned values.

        Returns
        -------
        y_hat : ndarray, shape (T,) or (T, D)
            Estimated values at integer points 0 ... T-1.
        spline : BSpline
            The BSpline object for further evaluation or differentiation.
        """
        if ctrl_y is None:
            if self._last_spline is None:
                raise RuntimeError("No previous fit() result and ctrl_y is None.")
            spline = self._last_spline
        else:
            ctrl_y = np.asarray(ctrl_y)
            if ctrl_y.shape[0] != self.n_ctrl:
                raise ValueError(f"ctrl_y must have length {self.n_ctrl} on axis 0")
            spline = BSpline(self.t, ctrl_y, self.k, extrapolate=False)

        y_hat = spline(self.x).astype(dtype)
        return y_hat, spline

    
    def fit_batch(self, y_batch: np.ndarray):
        """
        y_batch : ndarray, shape (B, T, D)    
        return
        -------
        ctrl_y_batch : ndarray, shape (B, n_ctrl, D), dtype float32
        """
        y_batch = np.asarray(y_batch, dtype=float)
        if y_batch.ndim == 2:          # (B, T) → (B, T, 1)
            y_batch = y_batch[..., None]

        if y_batch.shape[1] != self.T:
            raise ValueError(f"time dimension must be {self.T} (axis 1)")

        B, _, D = y_batch.shape

        y_perm = np.transpose(y_batch, (1, 0, 2))       # (T, B, D)
        y_2d = y_perm.reshape(self.T, B * D)            # (T, B*D)

        spline = make_lsq_spline(self.x, y_2d, self.t, self.k, axis=0)

        c = spline.c.reshape(self.n_ctrl, B, D)          # (n_ctrl, B, D)
        ctrl_y_batch = np.transpose(c, (1, 0, 2))        # (B, n_ctrl, D)

        ctrl_y_batch = np.ascontiguousarray(ctrl_y_batch, dtype=np.float32)
        self._last_spline = spline

        return ctrl_y_batch
    
        
    # ============== Local Refitting ==============c
    def refit_prefix(self,
                     y_prefix: np.ndarray,
                     ctrl_y:   np.ndarray,
                     n_prefix: int = 8,
                     n_free:   int = 3,
                     dtype=np.float32):
        """
        Adjusts only the first `n_free` control points to minimize squared error 
        over the first `n_prefix` points; remaining control points stay fixed.

        Parameters
        ----------
        y_prefix : ndarray, shape (n_prefix, D)
            Newly observed prefix segment.
        ctrl_y   : ndarray, shape (n_ctrl, D)
            Old control points (typically from a full fit).
        n_prefix : int, default 8
            Number of sampling points used for refitting.
        n_free   : int, default 3
            Number of control points to re-estimate (starting from index 0).
        dtype    : numpy dtype, default np.float32

        Returns
        -------
        new_ctrl_y : ndarray, shape (n_ctrl, D), dtype `dtype`
            Updated control points; only the first `n_free` points are modified.
        """
        y_prefix = np.asarray(y_prefix, dtype=float)
        ctrl_y   = np.asarray(ctrl_y,   dtype=float)

        if y_prefix.ndim == 1:                   # (n,) → (n,1)
            y_prefix = y_prefix[:, None]
        if ctrl_y.ndim == 1:
            ctrl_y = ctrl_y[:, None]

        n_p, D = y_prefix.shape
        if n_p != n_prefix:
            raise ValueError(f"y_prefix must have {n_prefix} rows (got {n_p}).")
        if ctrl_y.shape[0] != self.n_ctrl:
            raise ValueError(f"ctrl_y must have {self.n_ctrl} control points.")
        if n_free >= self.n_ctrl:
            raise ValueError("n_free must be less than total control-point number.")

        # -------- Φ (n_prefix × n_ctrl) --------
        x_pref = self.x[:n_prefix]
        Phi = np.empty((n_prefix, self.n_ctrl), float)
        for j in range(self.n_ctrl):
            coeff = np.zeros(self.n_ctrl); coeff[j] = 1.
            Phi[:, j] = BSpline(self.t, coeff, self.k)(x_pref)

        Phi_free  = Phi[:, :n_free]         # (n_prefix, n_free)
        Phi_fixed = Phi[:, n_free:]         # (n_prefix, n_fixed)

        # -------- Pseudo-inverse P = (ΦᵀΦ)⁻¹ Φᵀ --------
        P = np.linalg.pinv(Phi_free)        # (n_free, n_prefix)

        # -------- Contribution from fixed parts --------
        ctrl_fixed = ctrl_y[n_free:, :]     # (n_fixed, D)
        y_fixed    = Phi_fixed @ ctrl_fixed # (n_prefix, D)

        # -------- Update free control points --------
        residual  = y_prefix - y_fixed      # (n_prefix, D)
        ctrl_free = P @ residual            # (n_free,  D)

        new_ctrl = ctrl_y.copy()
        new_ctrl[:n_free, :] = ctrl_free
        return new_ctrl.astype(dtype)
        
    

    def refit_prefix_w(self,
                     y_prefix: np.ndarray,
                     ctrl_y:   np.ndarray,
                     n_prefix: int = 8,
                     n_free:   int = 3,
                     last_pt_weight: float = 0.0, # 新增：最后一个自由点的权重系数
                     dtype=np.float32):
        """
        Adjusts only the first `n_free` control points to minimize squared error 
        over the first `n_prefix` points; remaining control points stay fixed.

        Allows an additional penalty on the movement of the last free control point (index n_free-1).

        Parameters
        ----------
        y_prefix : ndarray, shape (n_prefix, D)
            Newly observed prefix segment.
        ctrl_y   : ndarray, shape (n_ctrl, D)
            Old control points.
        n_prefix : int, default 8
            Number of sampling points for refitting.
        n_free   : int, default 3
            Number of control points to re-estimate.
        last_pt_weight : float, default 0.0
            Penalty weight on the movement of the last free control point.
            Equivalent to original algorithm if 0. Larger values restrict adjustment of this point.
        dtype    : numpy dtype, default np.float32

        Returns
        -------
        new_ctrl_y : ndarray, shape (n_ctrl, D), dtype `dtype`
            Updated control points; only the first `n_free` points may be modified.
        """
        y_prefix = np.asarray(y_prefix, dtype=float)
        ctrl_y   = np.asarray(ctrl_y,   dtype=float)

        if y_prefix.ndim == 1:                   # (n,) → (n,1)
            y_prefix = y_prefix[:, None]
        if ctrl_y.ndim == 1:
            ctrl_y = ctrl_y[:, None]

        n_p, D = y_prefix.shape
        if n_p != n_prefix:
            raise ValueError(f"y_prefix must have {n_prefix} rows (got {n_p}).")
        if ctrl_y.shape[0] != self.n_ctrl:
            raise ValueError(f"ctrl_y must have {self.n_ctrl} control points.")
        if n_free > self.n_ctrl or n_free <= 0:
            raise ValueError("n_free must be positive and less than or equal to total control-point number.")

        # -------- Φ (n_prefix × n_ctrl) --------
        if len(self.x) < n_prefix:
             raise ValueError(f"Internal sample points `self.x` (len={len(self.x)}) is not long enough for `n_prefix`={n_prefix}.")
        x_pref = self.x[:n_prefix]
        
        Phi = np.empty((n_prefix, self.n_ctrl), float)
        for j in range(self.n_ctrl):
            coeff = np.zeros(self.n_ctrl); coeff[j] = 1.
            Phi[:, j] = BSpline(self.t, coeff, self.k)(x_pref)

        Phi_free  = Phi[:, :n_free]         # (n_prefix, n_free)
        Phi_fixed = Phi[:, n_free:]         # (n_prefix, n_fixed)

        # -------- Calculate fixed part contribution and residual --------
        ctrl_fixed = ctrl_y[n_free:, :]     # (n_fixed, D)
        y_fixed    = Phi_fixed @ ctrl_fixed # (n_prefix, D)
        residual   = y_prefix - y_fixed     # (n_prefix, D)

        # -------- Construct augmented least-squares system ------c
        if last_pt_weight > 1e-9 and n_free > 0:
            sqrt_w = np.sqrt(last_pt_weight)

            # Construct augmented matrix Phi_augc
            penalty_row = np.zeros((1, n_free), dtype=float)
            penalty_row[0, n_free - 1] = sqrt_w
            Phi_aug = np.vstack([Phi_free, penalty_row]) # shape: (n_prefix + 1, n_free)

            # Construct augmented target target_aug
            c_last_old = ctrl_y[n_free - 1, :] # (D,) or (1, D)
            penalty_target = sqrt_w * c_last_old
            target_aug = np.vstack([residual, penalty_target]) # shape: (n_prefix + 1, D)
            
            # Solve using the augmented system
            P_aug = np.linalg.pinv(Phi_aug)  # shape: (n_free, n_prefix + 1)
            ctrl_free = P_aug @ target_aug   # shape: (n_free, D)

        else: 
            P = np.linalg.pinv(Phi_free)     # (n_free, n_prefix)
            ctrl_free = P @ residual         # (n_free,  D)

        new_ctrl = ctrl_y.copy()
        new_ctrl[:n_free, :] = ctrl_free
        return new_ctrl.astype(dtype)
    

    @property
    def control_points(self):
        return self.ctrl_x




