# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import glob
import json
import os
import time

import eval_util
import losses
import frame_level_models
import video_level_models
import ensemble_model
import readers
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import tensorflow.contrib.slim as slim
import utils

import numpy as np

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to load the model files from. "
                      "The tensorboard metrics files are also saved to this "
                      "directory.")
  flags.DEFINE_string(
      "eval_data_config", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")

  # Other flags.
  flags.DEFINE_integer("batch_size", 1000,
                       "How many examples to process per batch.")
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")
  flags.DEFINE_boolean("run_once", True, "Whether to run eval only once.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")
  
  flags.DEFINE_string("feature_names", "predictions", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "3862", "Length of the feature vectors.")
  
  flags.DEFINE_string("output_dir", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string("prediction_feature_sizes", "3862", "Length of the feature vectors.")
  flags.DEFINE_integer("file_size", 800,
                       "Number of frames per batch for DBoF.")
  flags.DEFINE_integer("file_num_mod", None,
                       "file_num % 3 == file_num_mod will be output.")


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]
    return tf.train.batch_join(
        eval_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def build_graph(all_readers,
                all_eval_data_patterns, 
                model,
                label_loss_fn,
                batch_size=1024,
                num_readers=1):
  """Creates the Tensorflow graph for evaluation.

  Args:
    allreaders: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  # video_id_batch, model_input_raw, labels_batch, num_frames = get_input_evaluation_tensors(  # pylint: disable=g-line-too-long
  #     reader,
  #     eval_data_pattern,
  #     batch_size=batch_size,
  #     num_readers=num_readers)
  # tf.summary.histogram("model_input_raw", model_input_raw)

  # feature_dim = len(model_input_raw.get_shape()) - 1
  
  model_input_raw_tensors = []
  labels_batch_tensor = None
  video_id_batch = None
  
  for reader, data_pattern in zip(all_readers, all_eval_data_patterns):
    unused_video_id, model_input_raw, labels_batch, num_frames = (
        get_input_evaluation_tensors(
            reader,
            data_pattern,
            batch_size=batch_size))
    if labels_batch_tensor is None:
      labels_batch_tensor = labels_batch
    if video_id_batch is None:
      video_id_batch = unused_video_id
    model_input_raw_tensors.append(tf.expand_dims(model_input_raw, axis=2))
  
  model_input = tf.concat(model_input_raw_tensors, axis=2)
  labels_batch = labels_batch_tensor
    
  with tf.variable_scope("tower"):
    result = model.create_model(model_input,
                                num_frames=num_frames,
                                vocab_size=reader.num_classes,
                                labels=labels_batch,
                                is_training=False)
    predictions = result["predictions"]
    tf.summary.histogram("model_activations", predictions)
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  # tf.add_to_collection("input_batch_raw", model_input_raw)
  tf.add_to_collection("video_id_batch", video_id_batch)
  tf.add_to_collection("num_frames", num_frames)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("summary_op", tf.summary.merge_all())


def get_latest_checkpoint():
  index_files = file_io.get_matching_files(os.path.join(FLAGS.train_dir, 'model.ckpt-*.index'))

  # No files
  if not index_files:
    return None


  # Index file path with the maximum step size.
  latest_index_file = sorted(
      [(int(os.path.basename(f).split("-")[-1].split(".")[0]), f)
       for f in index_files])[-1][1]

  # Chop off .index suffix and return
  return latest_index_file[:-6]


def evaluation_loop(video_id_batch, prediction_batch, label_batch, loss,
                    summary_op, saver, summary_writer, evl_metrics,
                    last_global_step_val):
  """Run the evaluation loop once.

  Args:
    video_id_batch: a tensor of video ids mini-batch.
    prediction_batch: a tensor of predictions mini-batch.
    label_batch: a tensor of label_batch mini-batch.
    loss: a tensor of loss for the examples in the mini-batch.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.

  Returns:
    The global_step used in the latest model.
  """

  global_step_val = -1
  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
#     latest_checkpoint = get_latest_checkpoint()
#     if latest_checkpoint:
#       logging.info("Loading checkpoint for eval: " + latest_checkpoint)
#       # Restores from checkpoint
#       saver.restore(sess, latest_checkpoint)
#       # Assuming model_checkpoint_path looks something like:
#       # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
#       global_step_val = os.path.basename(latest_checkpoint).split("-")[-1]

#       # Save model
#       saver.save(sess, os.path.join(FLAGS.train_dir, "inference_model"))
#     else:
#       logging.info("No checkpoint file found.")
#       return global_step_val

#     if global_step_val == last_global_step_val:
#       logging.info("skip this checkpoint global_step_val=%s "
#                    "(same as the previous one).", global_step_val)
#       return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [video_id_batch, prediction_batch, label_batch, loss, summary_op]
    coord = tf.train.Coordinator()
    
    
    # output results
    start_time = time.time()
    video_ids = []
    video_labels = []
    video_features = []
    filenum = 0
    num_examples_processed = 0
    total_num_examples_processed = 0
    
    # output prediction dir
    directory = FLAGS.output_dir
    if directory != '':
      if not os.path.exists(directory):
          os.makedirs(directory)
      else:
          raise IOError("Output path exists! path='" + directory + "'")
    
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      evl_metrics.clear()

      examples_processed = 0
      while not coord.should_stop():
        batch_start_time = time.time()
        ids_val, predictions_val, labels_val, loss_val, summary_val = sess.run(
            fetches)
        seconds_per_batch = time.time() - batch_start_time
        example_per_second = labels_val.shape[0] / seconds_per_batch
        examples_processed += labels_val.shape[0]

        iteration_info_dict = evl_metrics.accumulate(predictions_val,
                                                     labels_val, loss_val)
        iteration_info_dict["examples_per_second"] = example_per_second

        iterinfo = utils.AddGlobalStepSummary(
            summary_writer,
            global_step_val,
            iteration_info_dict,
            summary_scope="Eval")
        logging.info("examples_processed: %d | %s", examples_processed,
                     iterinfo)
        
        # save predictions
        if directory != '':
          video_ids.append(ids_val)
          video_labels.append(labels_val)
          video_features.append(predictions_val)
          num_examples_processed += len(ids_val)

          if num_examples_processed >= FLAGS.file_size:
            assert num_examples_processed==FLAGS.file_size, "num_examples_processed should be equal to %d"%FLAGS.file_size
            video_ids = np.concatenate(video_ids, axis=0)
            video_labels = np.concatenate(video_labels, axis=0)
            video_features = np.concatenate(video_features, axis=0)
            write_to_record(video_ids, video_labels, video_features, filenum, num_examples_processed)

            video_ids = []
            video_labels = []
            video_features = []
            filenum += 1
            total_num_examples_processed += num_examples_processed

            now = time.time()
            logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
            num_examples_processed = 0

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      # calculate the metrics for the entire epoch
      epoch_info_dict = evl_metrics.get()
      epoch_info_dict["epoch_id"] = global_step_val

      summary_writer.add_summary(summary_val, global_step_val)
      epochinfo = utils.AddEpochSummary(
          summary_writer,
          global_step_val,
          epoch_info_dict,
          summary_scope="Eval")
      logging.info(epochinfo)
      evl_metrics.clear()
      
      # save prediction
      if directory != '':
        # if ids_val is not None:
        #   video_ids.append(ids_val)
        #   video_labels.append(labels_val)
        #   video_features.append(predictions_val)
        #   num_examples_processed += len(ids_val)

        if 0 < num_examples_processed <= FLAGS.file_size:
          video_ids = np.concatenate(video_ids, axis=0)
          video_labels = np.concatenate(video_labels, axis=0)
          video_features = np.concatenate(video_features, axis=0)
          write_to_record(video_ids, video_labels, video_features, filenum, num_examples_processed)
          total_num_examples_processed += num_examples_processed

          now = time.time()
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
          num_examples_processed = 0

        logging.info("Done with inference. %d samples was written to %s" % (total_num_examples_processed, FLAGS.output_dir))
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step_val


def evaluate():
  tf.set_random_seed(0)  # for reproducibility

  # Write json of flags
  # model_flags_path = os.path.join(FLAGS.train_dir, "model_flags.json")
  # if not file_io.file_exists(model_flags_path):
  #   raise IOError(("Cannot find file %s. Did you run train.py on the same "
  #                  "--train_dir?") % model_flags_path)
  # flags_dict = json.loads(file_io.FileIO(model_flags_path, mode="r").read())
  all_eval_data_patterns = []
  with open(FLAGS.eval_data_config) as f:
    all_eval_data_patterns = f.read().splitlines()

  with tf.Graph().as_default():
    # convert feature_names and feature_sizes to lists of values
    # feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
    #     flags_dict["feature_names"], flags_dict["feature_sizes"])

    # prepare a reader for each single model prediction result
    all_readers = []

    for i in xrange(len(all_eval_data_patterns)):
      reader = readers.EnsembleReader(
          feature_names=[FLAGS.feature_names], feature_sizes=[FLAGS.feature_sizes])
      all_readers.append(reader)

    input_reader = None
    input_data_pattern = None
    
    # model = find_class_by_name(flags_dict["model"],
    #     [frame_level_models, video_level_models])()
    model = ensemble_model.MeanModel()
    label_loss_fn = find_class_by_name("CrossEntropyLoss", [losses])()

    build_graph(
        all_readers=all_readers,
        all_eval_data_patterns = all_eval_data_patterns, 
        model=model,
        label_loss_fn=label_loss_fn,
        num_readers=FLAGS.num_readers,
        batch_size=FLAGS.batch_size)
    
    logging.info("built evaluation graph")
    video_id_batch = tf.get_collection("video_id_batch")[0]
    prediction_batch = tf.get_collection("predictions")[0]
    label_batch = tf.get_collection("labels")[0]
    loss = tf.get_collection("loss")[0]
    summary_op = tf.get_collection("summary_op")[0]

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir, graph=tf.get_default_graph())

    evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, FLAGS.top_k)

    last_global_step_val = -1
    while True:
      last_global_step_val = evaluation_loop(video_id_batch, prediction_batch,
                                             label_batch, loss, summary_op,
                                             saver, summary_writer, evl_metrics,
                                             last_global_step_val)
      if FLAGS.run_once:
        break

def write_to_record(video_ids, video_labels, video_features, filenum, num_examples_processed):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir + '/' + 'predictions-%04d.tfrecord' % filenum)
    for i in range(num_examples_processed):
        video_id = video_ids[i]
        video_label = np.nonzero(video_labels[i,:])[0]
        example = get_output_feature(video_id, video_label, [video_features[i,:]], ['predictions'])
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def get_output_feature(video_id, video_label, video_feature, feature_names):
    feature_maps = {'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=video_label))}
    for feature_index in range(len(feature_names)):
        feature_maps[feature_names[feature_index]] = tf.train.Feature(
            float_list=tf.train.FloatList(value=video_feature[feature_index]))
    example = tf.train.Example(features=tf.train.Features(feature=feature_maps))
    return example
  
def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  evaluate()


if __name__ == "__main__":
  app.run()
