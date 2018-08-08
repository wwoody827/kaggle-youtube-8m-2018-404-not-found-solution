data_config="ensemble/ensemble_val_on_val.config"

for train_dir in "${train_dirs[@]}"
do
  file=/media/woody/SSD/home/woody/val/validate0*.tfrecord
  CUDA_VISIBLE_DEVICES=1 python inference_prediction_val.py --input_data_pattern=$file  --train_dir="run/$train_dir" --batch_size=50 --output_dir="/media/woody/Data2T/youtube-8m/prediction/prediction_val_0/$train_dir" --file_size=4000                                                            
done
