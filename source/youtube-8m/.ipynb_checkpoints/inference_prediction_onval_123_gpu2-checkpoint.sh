train_dirs=(
            # "0620_gatednetvladLF-256-1024-80-0002-300iter-norelu-basic-gatedmoe" 
            "0624_gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe"
            "0627_gatednetvladLF-128-1024-80-0002-4chain-300iter-norelu-basic-DeepCombineChainModel"
            "0628_gatednetvladLF-128-1024-40-0002-4chain-256cells-300iter-norelu-basic-DeepCombineChainModel"
            )

for train_dir in "${train_dirs[@]}"
do
  file=/media/woody/Woody/Data/youtube-8m/frame/val/validate[123]*.tfrecord
  CUDA_VISIBLE_DEVICES=1 python inference_prediction_val.py --input_data_pattern=$file  --train_dir="run/$train_dir" --batch_size=50 --output_dir="/media/woody/Data2T/youtube-8m/prediction/prediction_run2_val_123/$train_dir" --file_size=4000
done
