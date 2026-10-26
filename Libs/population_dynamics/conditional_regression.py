"""Cell-conditioned ridge and RBF regression for whole-block state probes.

Extracted without numerical changes from state_representation_probe.py.
All centering, active-feature selection and scaling use fitting rows only.
"""
from __future__ import annotations

import numpy as np


def conditional_prediction(train_x,train_y,train_cells,test_x,test_cells,kind,alpha):
    if not np.isfinite(alpha) or alpha<=0:raise ValueError('positive finite regularization required')
    unique=np.unique(train_cells);n=len(train_x)
    if not set(test_cells).issubset(set(unique)):raise ValueError('unseen factorial cell')
    c=(train_cells[:,None]==unique).astype(float);ct=(test_cells[:,None]==unique).astype(float)
    inv=np.diag(1/c.sum(0))
    cell_mean=np.asarray([train_x[train_cells==cell].mean(0) for cell in unique]);xc=train_x-c@cell_mean
    # Fractional cell weights can leave roundoff residuals in a cell-constant
    # feature. Such a feature contains no state variation and must stay inactive.
    active=np.any(np.asarray([np.ptp(train_x[train_cells==cell],axis=0)>0 for cell in unique]),axis=0)
    scale=np.std(xc,axis=0)
    if np.any(~np.isfinite(scale[active])) or np.any(scale[active]<=0):raise ValueError('unresolved training feature scale')
    z=xc[:,active]/scale[active];zt=(test_x-ct@cell_mean)[:,active]/scale[active]
    ymean=np.asarray([train_y[train_cells==cell].mean(0) for cell in unique])
    if kind=='cells' or not np.any(active):return ct@ymean,dict(active_features=int(active.sum()),effective_df=float(len(unique)))
    projection=np.eye(n)-c@inv@c.T
    if kind=='linear':kernel=z@z.T;cross=zt@z.T
    elif kind=='rbf':
        def gaussian(a,b):
            distances=np.maximum(0,np.sum(a*a,axis=1)[:,None]+np.sum(b*b,axis=1)[None,:]-2*a@b.T)
            return np.exp(-distances/(2*z.shape[1]))
        kernel=gaussian(z,z);cross=gaussian(zt,z)
    else:raise ValueError('unknown conditional predictor')
    centered=projection@kernel@projection;test_centered=(cross-ct@inv@c.T@kernel)@projection
    matrix=centered+float(alpha)*n*np.eye(n);dual=np.linalg.solve(matrix,projection@train_y)
    df=float(np.trace(np.linalg.solve(matrix,centered))+len(unique))
    prediction=ct@ymean+test_centered@dual
    if not np.isfinite(prediction).all():raise ValueError('nonfinite conditional prediction')
    return prediction,dict(active_features=int(active.sum()),effective_df=df)

