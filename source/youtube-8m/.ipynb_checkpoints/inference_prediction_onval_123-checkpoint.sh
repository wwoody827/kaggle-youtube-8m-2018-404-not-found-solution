train_dirs=(
            # "0620_gatednetvladLF-256-1024-80-0002-300iter-norelu-basic-gatedmoe" 
            "0621_gatednetvladLF-128-1024-160-0002-300iter-norelu-basic-gatedmoe"
            "0622_gatednetfvLF-128k-1024-60-0002-300iter-norelu-basic-gatedmoe"
            "0622_gatednetvladLF-64-1024-160-0002-300iter-norelu-basic-gatedmoe"
            )

for train_dir in "${train_dirs[@]}"
do
  file=/media/woody/Woody/Data/youtube-8m/frame/val/validate[123]*.tfrecord
  CUDA_VISIBLE_DEVICES=0 python inference_prediction_val.py --input_data_pattern=$file  --train_dir="run/$train_dir" --batch_size=50 --output_dir="/media/woody/Data2T/youtube-8m/prediction/prediction_run2_val_123/$train_dir" --file_size=4000
done
