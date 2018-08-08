train_dirs=(# "0620_gatednetvladLF-256-1024-80-0002-300iter-norelu-basic-gatedmoe" 
            # "0621_gatednetvladLF-128-1024-160-0002-300iter-norelu-basic-gatedmoe"
            # "0622_gatednetfvLF-128k-1024-60-0002-300iter-norelu-basic-gatedmoe"
            # "0622_gatednetvladLF-64-1024-160-0002-300iter-norelu-basic-gatedmoe"
            # "0705_distill_gatednetvladLF-128-1024-160-0002-300iter-norelu-basic-gatedmoe_180477"
            # "0705_distill_gatednetvladLF-128-1024-160-0002-300iter-norelu-basic-gatedmoe_240981"
            # "0707_distill_0.75_gatednetvladLF-128-1024-160-0002-300iter-norelu-basic-gatedmoe"
            "0707_distill_0.99_gatednetvladLF-128-1024-160-0002-300iter-norelu-basic-gatedmoe"
            "0709_distill_0.99_gatednetvladLF-210-1024-160-0002-300iter-norelu-basic-gatedmoe"
            )

for train_dir in "${train_dirs[@]}"
do
  for file in /media/woody/Woody/Data/youtube-8m/frame/train/train*.tfrecord
  do
    CUDA_VISIBLE_DEVICES=0 python inference_prediction.py --input_data_pattern=$file  --train_dir="run/$train_dir" --batch_size=100 --output_dir="/media/woody/Woody/Data/youtube-8m/prediction_train/$train_dir" --file_size=10000                                                            
  done
done