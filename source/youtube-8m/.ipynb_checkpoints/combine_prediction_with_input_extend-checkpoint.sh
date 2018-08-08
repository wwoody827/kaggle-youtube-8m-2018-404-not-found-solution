#!/bin/bash  
train_folder="/media/woody/Woody/Data/youtube-8m/frame/val/"
prediction_folder="/media/woody/Data2T/youtube-8m/prediction/prediction_train_ensemble/0704_MoeModel_1_extend/"
output_dir="/media/woody/Data2T/youtube-8m/prediction/train_distill_extend"
files="validate[123]*.tfrecord"
# for file in "$prediction_folder$files"
for f in $prediction_folder$files
do
	#python combine_prediction_with_input.py --input_data_pattern=$file --prediction_data_pattern=/train0001_predictions.tfrecord  --output_dir=prediction_train_input/   
    echo "Processing $f file..."
    basename=$(basename $f)
    original=${basename::-30}".tfrecord"
    python combine_prediction_with_input.py --input_data_pattern=$train_folder$original --prediction_data_pattern=$f  --output_dir=$output_dir
done