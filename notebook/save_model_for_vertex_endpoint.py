#%%
import os
import sys

local_path = os.getcwd()
src_path = os.sep.join(local_path.split(os.sep)[:-1])
sys.path.append(src_path)

from demo_lib import *


# Load model and train data
model, y_train = load_model()

# Dump it to checkpoint for creating the saved_model
checkpoint_dir = os.path.join('..', 'bin', 'model_checkpoint')
checkpoint_path = save_model_to_checkpoint(model, checkpoint_dir)

del model

# Create the wrapped model to save
wrapper_model = TFPModelWrapper(build_model,
                                y_train,
                                last_train_day="2023-10-31",
                                trend=False,
                                variational_posteriors_samples=10000)

# Update the model parameters with those from the train
wrapper_model.load_parameters(checkpoint_path)

# Save model to ../bin/saved_model
export_dir = os.path.join('..', 'bin', 'saved_model')
wrapper_model.save(export_dir)

