"""
Copyright (c) 2025 University of Southern California
See full notice in [LICENSE.md](LICENSE.md)
Parsa Vahidi, Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""


"""The model used in BRAID"""
"""For mathematical descriptions see BRAIDModelDoc.md"""

import copy
import io
import logging
import os
import re
import time
import warnings
from datetime import datetime
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .MainModel import MainModel, shift_ms_to_1s_series
from .tools.abstract_classes import PredictorModel
from .tools.tf_losses import (
    masked_CategoricalCrossentropy,
    masked_CC,
    masked_mse,
    masked_PoissonLL_loss,
    masked_R2,
)
from .tools.tools import applyScaling, transposeIf, undoScaling

logger = logging.getLogger(__name__)



# PredictorModel
class BRAIDModel(MainModel):
    """The main class implementing BRAID. 
    x1(k+1) = A1( x1(k) ) + K1( y(k), u(k) )
    x2(k+1) = A2( x2(k) ) + K2( x1(k+1), y(k), u(k) )
    x3(k+1) = A3( x3(k) ) + K3( u(k) )
    y(k)    = Cy( x1(k), x2(k), u(k) ) + ey_k
    z(k)    = Cz( x1(k), x2(k), x3(k), u(k) ) + ez_k
    x(k) = [x1(k); x2(k); x3(k)] => Latent state time series
    x1(k) => Latent states related to z
    x2(k) => Latent states unrelated to z 
    x3(k) => Latent states unrelated to y 
    """   
     
    def __init__(self, 
            block_samples=128,   # RNN sequence length
            batch_size=32,       # Training batch size
            log_dir = '', # If not empty, will store tensorboard logs
            missing_marker = None, # Values of z that are equal to this will not be used. Set this to None as not supported.
            **kwargs):             # Pass other settings define in the parent object (MainModel) as a dictionary 
        
        self.block_samples = block_samples
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.missing_marker = missing_marker

        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, Y, Z=None, U=None, nx=None, n1=None, n3=None, n_pre=None, # data should be passed as dim * time
            create_val_from_training = False, # If True, will create the validation data by cutting it off of the training data
            validation_set_ratio = 0.2, # Ratio of training data to cut off to make the validation data 
            Y_validation = None, # if provided will use to compute loss on validation (unless create_val_from_training=True)
            Z_validation = None, # if provided will use to compute loss on validation (unless create_val_from_training=True)
            U_validation = None, # if provided will use to compute loss on validation (unless create_val_from_training=True)
            true_model = None,
            YType=None, ZType=None, UType=None,
            steps_ahead=None, # List of ints (None take as [1]), indicating the number of steps ahead to generate from the model (used to construct training loss and predictions
            A_args = {}, K_args = {}, Cy_args = {}, Cz_args = {},    # Both stage 1 and 2 params
            A1_args = None, K1_args = None, Cy1_args = None, Cz1_args = None, # Stage 1 params
            A2_args = None, K2_args = None, Cy2_args = None, Cz2_args = None, # Stage 2 params
            A3_args = None, K3_args = None, Cy3_args = None, Cz3_args = None, # Stage 2 params,
            args_base = None,
            noFT = False,
            isFullyLinear=False
        ): 

        if args_base is None:
            raise Exception("args_base should be provided externally to BRAIDModel fitting. It was None!")
        self.steps_ahead = args_base['steps_ahead']

        ny, Ndat = Y.shape[0], Y.shape[1]
        if Z is not None:
            nz, NdatZ = Z.shape[0], Z.shape[1]
        else:
            nz = 0
        if U is not None:
            nu = U.shape[0]
        else:
            nu = 0
        
        if n_pre is not None and n_pre > 0:
            ### Preprocessing stage: fitting a MainModel with stage 2 alone
            args_pre = copy.deepcopy(args_base)
            args_pre['model2_Cz_Full']=False
            args_pre['allow_nonzero_Cz2']=True
            args_pre['has_UFT_reg']=False
            sId_pre = MainModel(log_dir=self.log_dir,
                    missing_marker=self.missing_marker)
            sId_pre.fit(Y, Z, U=U, nx=n_pre, n1=0,
                    YType=YType, ZType=ZType, 
                    Y_validation=Y_validation, Z_validation=Z_validation, U_validation=U_validation,
                    **args_pre)
        
            zPredRes1Train, _, _, _, _, _ = runPredict(sId_pre, Y=transposeIf(Y), Z=transposeIf(Z), U=transposeIf(U), YType=YType, ZType=ZType, useXFilt=False, missing_marker=self.missing_marker)
            if Y_validation is not None:
                zPredRes1Val, _, _, _, _, _ = runPredict(sId_pre, Y=transposeIf(Y_validation), Z=transposeIf(Z_validation), U=transposeIf(U_validation), YType=YType, ZType=ZType, useXFilt=False, missing_marker=self.missing_marker)
            else:
                zPredRes1Val = None

            if isinstance(zPredRes1Train, list):
                zPredRes1Train = zPredRes1Train[0]
            if isinstance(zPredRes1Val, list):
                zPredRes1Val = zPredRes1Val[0]
            
        else:
            sId_pre = None
            zPredRes1Train = transposeIf(Z)
            zPredRes1Val = transposeIf(Z_validation)
        if n3 is None:
            n3 = 0
        if nx > n1+n3: # n1+n3<nx => fit d(x1):n1, d(x2):nx-n1-n3>0, d(x3):n3
            nxThis = nx - n3
            n1This = n1
            n3This = n3
            n2This = nx - n1This - n3This
        elif nx > n1: # n1<nx<=n1+n3 => fit d(x1):n1, d(x2):0, d(x3):nx-n1
            nxThis = n1
            n1This = n1
            n3This = nx - n1
            n2This = 0
        else: # nx<=n1 => fit d(x1):nx, d(x2):0, d(x3):0
            nxThis = nx
            n1This = nx
            n3This = 0
            n2This = 0

        ### Stages 1,2 fitted as a MainModel
        args = copy.deepcopy(args_base)
        args['model2_Cz_Full']=False
        args['allow_nonzero_Cz2']=False 
        if noFT:
            args['has_UFT']=False
            args['has_UFT_reg']=False
        sId = MainModel(log_dir=self.log_dir,
                missing_marker=self.missing_marker)
        sId.fit(Y, transposeIf(zPredRes1Train), U=U, nx=nxThis, n1=n1This, 
                YType=YType, ZType=ZType, 
                Y_validation=Y_validation, Z_validation=transposeIf(zPredRes1Val), U_validation=U_validation, # This is not generally useful
                true_model=true_model, 
                **args)

       
        if n3This>0 and nu>0:
            # Stage 3
            args_post = copy.deepcopy(args_base)
            args_post['skip_Cy']=True 
            args_post['allow_nonzero_Cz2']=False 
            zPredRes2Train, _, _, _, _, _ = runPredict(sId, Y=transposeIf(Y), Z=zPredRes1Train, U=transposeIf(U), YType=YType, ZType=ZType, useXFilt=False, missing_marker=self.missing_marker)
            if U_validation is not None:                
                zPredRes2Val, _, _, _, _, _ = runPredict(sId, Y=transposeIf(Y_validation), Z=zPredRes1Val, U=transposeIf(U_validation), YType=YType, ZType=ZType, useXFilt=False, missing_marker=self.missing_marker)
            else:
                zPredRes2Val = None

            if isinstance(zPredRes2Train, list):
                zPredRes2Train = zPredRes2Train[0]
            if isinstance(zPredRes2Val, list):
                zPredRes2Val = zPredRes2Val[0]

            Z3 = Z - transposeIf(zPredRes2Train)
            if Z_validation is not None:
                Z3_validation = Z_validation - transposeIf(zPredRes2Val)  
            else:
                Z3_validation = None

            sId_post = MainModel(log_dir=self.log_dir,
                    missing_marker=self.missing_marker)

            sId_post.fit(None, Z3, U=U, nx=n3This, n1=n3This, 
                    YType=YType, ZType=ZType, 
                    Y_validation=None, Z_validation=Z3_validation, U_validation=U_validation,
                    **args_post)
        else:
            sId_post = None

        self.sId_pre = sId_pre
        self.sId = sId
        self.sId_post = sId_post
        self.nx = nx
        self.n1 = n1This
        self.n3 = n3This
        self.n2 = n2This
        self.ny = ny
        self.nz = nz
        self.nu = nu

        self.blown_up = self.hasBlownUp()

   
    def hasBlownUp(self):
        blown_up = False
        if hasattr(self, 'sId') and  hasattr(self.sId, 'blown_up') and self.sId.blown_up:
            blown_up = True
        if hasattr(self, 'sId_pre') and  hasattr(self.sId_pre, 'blown_up') and self.sId_pre.blown_up:
            blown_up = True
        if hasattr(self, 'sId_post') and  hasattr(self.sId_post, 'blown_up') and self.sId_post.blown_up:
            blown_up = True
        return blown_up


    def discardModels(self):
        """Prepares the object for pickling by replacing tf models with 
        dictionaries of their weights
        """        
        if hasattr(self, 'sId_pre') and hasattr(self.sId_pre, 'discardModels'):
            self.sId_pre.discardModels()
        if hasattr(self, 'sId') and hasattr(self.sId, 'discardModels'):
            self.sId.discardModels()
        if hasattr(self, 'sId_post') and hasattr(self.sId_post, 'discardModels'):
            self.sId_post.discardModels()


    def restoreModels(self):
        """Prepares the object for use after loading from a pickled file 
        by creating tf models and populating them with the saved weights
        """
        if hasattr(self, 'sId_pre') and hasattr(self.sId_pre, 'restoreModels'):
            self.sId_pre.restoreModels()
        if hasattr(self, 'sId') and hasattr(self.sId, 'restoreModels'):
            self.sId.restoreModels()
        if hasattr(self, 'sId_post') and hasattr(self.sId_post, 'restoreModels'):
            self.sId_post.restoreModels()

    def getLSSM(self): 
        return self.sId.getLSSM()


    def set_multi_step_with_data_gen(self, multi_step_with_data_gen, update_rnn_model_steps=True, noise_samples=0):
        self.sId.set_multi_step_with_data_gen(multi_step_with_data_gen, update_rnn_model_steps=update_rnn_model_steps, noise_samples=noise_samples)
        if self.n3 > 0 and hasattr(self, self.sId_post):
            self.sId_post.set_multi_step_with_data_gen(multi_step_with_data_gen, update_rnn_model_steps=update_rnn_model_steps, noise_samples=noise_samples)

    def set_steps_ahead(self, steps_ahead, update_rnn_model_steps=True):
        self.sId.set_steps_ahead(steps_ahead, update_rnn_model_steps=update_rnn_model_steps)
        if self.n3 > 0 and hasattr(self, self.sId_post):
            self.sId_post.set_steps_ahead(steps_ahead, update_rnn_model_steps=update_rnn_model_steps)
        if hasattr(self, 'steps_ahead'):
            self.steps_ahead = steps_ahead


    def predict(self, Y, U=None, x0=None):
        
        Ndat = Y.shape[0]

        steps_ahead = self.steps_ahead if hasattr(self, 'steps_ahead') and self.steps_ahead is not None else [1]
        steps_ahead, _, steps_ahead_model1, _, model1_orig_step_inds \
             = self.sId.get_model_steps_ahead(steps_ahead)
        allXp_steps = [np.zeros((Ndat, self.nx)) for s in steps_ahead]
        allYp_steps = [None for s in steps_ahead]
        allZp_steps = [None for s in steps_ahead]

        additionalArgs = {}
        preds = self.sId.predict(Y, U, **additionalArgs)
        allZp_steps = list(preds[                  :  len(steps_ahead)])
        allYp_steps = preds[  len(steps_ahead):2*len(steps_ahead)]
        allXp12_steps = preds[2*len(steps_ahead):3*len(steps_ahead)]

        for saInd in range(len(steps_ahead)):
            allXp_steps[saInd][:, :self.n1+self.n2] = allXp12_steps[saInd]

        if self.n3 > 0 and hasattr(self, 'sId_post') and self.sId_post is not None:
            preds_post = self.sId_post.predict(None, U, **additionalArgs)
            allZp3_steps = preds_post[                  :  len(steps_ahead)]
            allXp3_steps = preds_post[2*len(steps_ahead):3*len(steps_ahead)]
            for saInd in range(len(steps_ahead)):
                allXp_steps[saInd][:, self.n1+self.n2:] = allXp3_steps[saInd]
                allZp_steps[saInd] = allZp_steps[saInd] + allZp3_steps[saInd]

       
        return tuple(allZp_steps) + tuple(allYp_steps) + tuple(allXp_steps)



def runPredWithArgs(args):
    return args[0].predict(args[1], args[2], **args[3])

def runPredict(sId, Y=None, Z=None, U=None, YType=None, ZType=None, useXFilt=False, missing_marker=None, undo_scaling=False):
    """Runs the model prediction after applying appropriate preprocessing (e.g. zscoring) on the input data and also 
    undoes the preprocessing (e.g. zscoring) in the predicted data.
    
    Args:
        sId (PredictorModel): a model that implements a predict method.
        Y (np.array): input data. Defaults to None.
        Z (np.array): output data. Defaults to None.
        U (np.array): external input data. Defaults to None.
        YType (string, optional): data type of Y. Defaults to None.
        ZType (string, optional): data type of Z. Defaults to None.
        useXFilt (bool, optional): if true, will pass to predict if the 
            model supports that argument (i.e. is an LSSM). Defaults to False.
        missing_marker (numpy value, optional): indicator of missing samples in data. 
            Is used in performing and undoing preprocessing. Defaults to None.
        undo_scaling (bool, optional): if true, will apply the inverse scaling 
            on predictions. Defaults to False.

    Returns:
        zPred (np.array): predicted Z (A list with each element containing prediction for one desired step-ahead; parameter self.steps_ahead)
        yPred (np.array): predicted Y (A list with each element containing prediction for one desired step-ahead; parameter self.steps_ahead)
        xPred (np.array): latent state X (A list with each element containing prediction for one desired step-ahead; parameter self.steps_ahead)
        Y (np.array): updated Y after any preprocessing/undoing
        Z (np.array): updated Z after any preprocessing/undoing
        U (np.array): updated U after any preprocessing/undoing
    """    
    # ‌Apply any necessary scaling
    if YType == 'cont':
        Y = applyScaling(sId, Y, 'yMean', 'yStd', missing_marker=missing_marker)
    if ZType == 'cont' and Z is not None:
        Z = applyScaling(sId, Z, 'zMean', 'zStd', missing_marker=missing_marker)
    if U is not None:
        U = applyScaling(sId, U, 'uMean', 'uStd', missing_marker=missing_marker)

    steps_ahead = [1]
    if hasattr(sId, 'steps_ahead') and sId.steps_ahead is not None:
        steps_ahead = sId.steps_ahead

    # Evaluate decoding on test data
    additionalArgs = {}
    if 'PSID.LSSM.LSSM' in str(type(sId)):
        additionalArgs['useXFilt'] = useXFilt
    if isinstance(Y, (list, tuple)):
        zPred, yPred, xPred = [], [], []
        argsAll = []
        for trialInd in range(len(Y)):
            argsAll.append( (sId, Y[trialInd], U[trialInd] if U is not None else None, additionalArgs) )


        results = []
        for args in argsAll:
            results.append( runPredWithArgs(args) )

        for trialInd in range(len(Y)):
            # predsThis = sId.predict(Y[trialInd], U[trialInd] if U is not None else None, **additionalArgs)
            predsThis = results[trialInd]
            zPredThis = list(shift_ms_to_1s_series( predsThis[                  :  len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
            yPredThis = list(shift_ms_to_1s_series( predsThis[  len(steps_ahead):2*len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
            xPredThis = list(shift_ms_to_1s_series( predsThis[2*len(steps_ahead):3*len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
            zPred.append(zPredThis)
            yPred.append(yPredThis)
            xPred.append(xPredThis)
    else:
        preds = sId.predict(Y, U, **additionalArgs)
        zPred = list(shift_ms_to_1s_series( preds[                  :  len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
        yPred = list(shift_ms_to_1s_series( preds[  len(steps_ahead):2*len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
        xPred = list(shift_ms_to_1s_series( preds[2*len(steps_ahead):3*len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))

        if hasattr(sId, 'zErrSys') and hasattr(sId.zErrSys, 'UInEps') and sId.zErrSys.UInEps:
            sId.zErrSys.K = 0 * sId.zErrSys.K 
            sId.zErrSys.B_KD = sId.zErrSys.B
            sId.zErrSys.A_KC = sId.zErrSys.A
            preds_eps = sId.zErrSys.predict(Y, U, **additionalArgs)
            zPred_eps = list(shift_ms_to_1s_series( preds_eps[                  :  len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
            for z_sta_ind, zPred_step in enumerate(zPred):
                zPred[z_sta_ind] += zPred_eps[z_sta_ind] 

    if undo_scaling:
        if YType == 'cont':
            yPred = undoScaling(sId, yPred, 'yMean', 'yStd', missing_marker=missing_marker)
            Y = undoScaling(sId, Y, 'yMean', 'yStd', missing_marker=missing_marker)
        if ZType == 'cont':
            zPred = undoScaling(sId, zPred, 'zMean', 'zStd', missing_marker=missing_marker)
            Z = undoScaling(sId, Z, 'zMean', 'zStd', missing_marker=missing_marker)
        if U is not None:
            U = undoScaling(sId, U, 'uMean', 'uStd', missing_marker=missing_marker)
    return zPred, yPred, xPred, Y, Z, U