import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json,time
import numpy as np
import tensorflow as tf
from pathlib import Path
started=time.perf_counter();tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();gpus=tf.config.list_physical_devices('GPU');assert gpus
sequence=tf.keras.Input((None,12),name='sequence');mask=tf.keras.Input((None,),dtype='bool',name='valid_mask');h=tf.keras.layers.GRU(64)(sequence,mask=mask);out=tf.keras.layers.Dense(1,activation='sigmoid')(h);model=tf.keras.Model({'sequence':sequence,'valid_mask':mask},out);model.compile(optimizer=tf.keras.optimizers.Adam(.0005),loss='binary_crossentropy')
rng=np.random.default_rng(42);x=rng.normal(size=(4,23,12)).astype('float32');lengths=np.array([2,9,15,23]);valid=np.arange(23)[None,:]<lengths[:,None];x[~valid]=0
loss=float(model.train_on_batch({'sequence':x,'valid_mask':valid},np.array([0,1,0,1],dtype='int32'),sample_weight=np.ones(4,dtype='float32')))
v=model.trainable_variables[0];device=v.value.handle.device if hasattr(v,'value') else v.handle.device
assert 'GPU:' in device,device
result={'status':'passed','tensorflow':tf.__version__,'gpu_devices':[d.name for d in gpus],'variable_device':device,'variable_length_masked_gru_deterministic_training':True,'elapsed_seconds':time.perf_counter()-started,'test_loss':loss}
Path('/content/seqcredit/artifacts/colab_gpu_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
