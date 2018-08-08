import sys
import os
import subprocess
from subprocess import call

data_config="ensemble/ensemble_infer_on_train.config"

train_dir = "run/ensemble/0704_MoeModel_1"
output_dir="/media/woody/Data2T/youtube-8m/prediction/prediction_train_ensemble/0704_MoeModel_1/"

with open(data_config) as f:
    all_data_patterns = f.read().splitlines()

data_file_list = os.listdir(all_data_patterns[0])
data_file_list.sort()

def write_list(items, filename):
  with open(filename, 'w') as f:
    for item in items:
      f.write("%s\n" % item)
  return

for data_file in data_file_list:
  data_file_list_new = [os.path.join(x, data_file) for x in all_data_patterns]
  tempfile = data_config + '.temp'
  write_list(data_file_list_new, tempfile)
  cmd = "CUDA_VISIBLE_DEVICES=\"0\" python inference_prediction_ensemble.py --data_config=" + tempfile + " --model=MoeModel --train_dir=" + train_dir + " --batch_size=1000 --output_dir=" + output_dir
  print(cmd)
  call(cmd, shell=True)
  
  