# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python script to generate TFRecords of SequenceExample from raw videos."""

import contextlib
import math
import os
from typing import Dict, Optional, Sequence

from absl import app
from absl import flags
import ffmpeg
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa

import warnings
warnings.filterwarnings("ignore")


flags.DEFINE_string("csv_path", None, "Input csv")
flags.DEFINE_string("output_path", None, "Tfrecords output path.")
flags.DEFINE_string("video_root_path", None,
                    "Root directory containing the raw videos.")
flags.DEFINE_integer(
    "num_shards", -1, "Number of shards to output, -1 means"
                      "it will automatically adapt to the sqrt(num_examples).")
flags.DEFINE_bool("decode_audio", False, "Whether or not to decode the audio")
flags.DEFINE_bool("shuffle_csv", False, "Whether or not to shuffle the csv.")
FLAGS = flags.FLAGS

_JPEG_HEADER = b"\xff\xd8"


@contextlib.contextmanager
def _close_on_exit(writers):
    """Call close on all writers on exit."""
    try:
        yield writers
    finally:
        for writer in writers:
            writer.close()


def chunk_spectro(spectrogram: np.ndarray, vid_length=10, window_samples: int=100):
    freq_bins, time_steps = spectrogram.shape
    final = np.zeros((time_steps//window_samples, freq_bins, window_samples)) # could have extra zeros in there?
    for t in range(vid_length):
        sample = t*window_samples
        chunk = spectrogram[:, sample:sample+window_samples]
        final[t] = chunk
    return final

def add_spectrogram(key: str, spectrogram: np.ndarray, sequence: tf.train.SequenceExample):
    # maybe it expects it in 1s chunks?
    fl_spectro = sequence.feature_lists.feature_list[key]
    # spectrogram = np.moveaxis(spectrogram, 0, -1)
    print(spectrogram.shape)
    chunked_spectro = chunk_spectro(spectrogram, vid_length=10, window_samples=100)
    # fl_spectro.feature.add().float_list.value[:] = spectrogram_flat
    for i in range(chunked_spectro.shape[0]):
        fl_spectro.feature.add().float_list.value[:] = chunked_spectro[i].flatten()
        # fl_spectro.feature.add().float_list.value[:] = spectrogram[i, :]
        # for f in spectrogram[i, :]:
        #     fl_spectro.feature.add().float_list.value[:] = f


def add_float_list(key: str, values: Sequence[float],
                   sequence: tf.train.SequenceExample):
    sequence.feature_lists.feature_list[key].feature.add(
    ).float_list.value[:] = values


def add_bytes_list(key: str, values: Sequence[bytes],
                   sequence: tf.train.SequenceExample):
    sequence.feature_lists.feature_list[key].feature.add(
    ).bytes_list.value[:] = values


def add_int_list(key: str, values: Sequence[int],
                 sequence: tf.train.SequenceExample):
    sequence.feature_lists.feature_list[key].feature.add(
    ).int64_list.value[:] = values


def set_context_int_list(key: str, value: Sequence[int],
                         sequence: tf.train.SequenceExample):
    sequence.context.feature[key].int64_list.value[:] = value


def set_context_bytes(key: str, value: bytes,
                      sequence: tf.train.SequenceExample):
    sequence.context.feature[key].bytes_list.value[:] = (value,)


def set_context_float(key: str, value: float,
                      sequence: tf.train.SequenceExample):
    sequence.context.feature[key].float_list.value[:] = (value,)


def set_context_int(key: str, value: int, sequence: tf.train.SequenceExample):
    sequence.context.feature[key].int64_list.value[:] = (value,)


def extract_frames(video_path: str,
                   start: float,
                   end: float,
                   fps: int = 10,
                   min_resize: int = 256):
    """Extract list of jpeg bytes from video_path using ffmpeg."""
    new_width = "(iw/min(iw,ih))*{}".format(min_resize)
    cmd = (
        ffmpeg
            .input(video_path)
            .trim(start=start, end=end)
            .filter("fps", fps=fps)
            .filter("scale", new_width, -1)
            .output("pipe:", format="image2pipe")
    )
    jpeg_bytes, _ = cmd.run(capture_stdout=True, quiet=True)
    jpeg_bytes = jpeg_bytes.split(_JPEG_HEADER)[1:]
    jpeg_bytes = map(lambda x: _JPEG_HEADER + x, jpeg_bytes)
    return list(jpeg_bytes)


def extract_audio(video_path: str,
                  start: float,
                  end: float,
                  sampling_rate: int = 48000):
    """Extract raw mono audio float list from video_path with ffmpeg."""
    cmd = (
        ffmpeg
            .input(video_path, ss=start, t=end - start)
            .output("pipe:", ac=1, ar=sampling_rate, format="s32le")
    )
    audio, _ = cmd.run(capture_stdout=True, quiet=True)
    audio = np.frombuffer(audio, np.float32)
    return list(audio)


def extract_spectro(video_path: str,
                    start: float,
                    end: float,
                    sampling_rate: int = 16000):
    audio, sr = librosa.load(video_path, offset=start, duration=(end - start), sr=sampling_rate)
    w_len = int(sampling_rate/1000 * 25) #25ms window length
    h_len = int(sampling_rate/1000 * 10) #10ms hop length

    S = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=128, hop_length=h_len, win_length=w_len)

    # Clamp the time dimension so it is a multiple of 100
    _, time_steps = S.shape
    time_steps = time_steps - (time_steps % 100)
    S = S[:, 0:time_steps]
    return S


def generate_sequence_example(video_path: str,
                              start: float,
                              end: float,
                              label_name: Optional[str] = None,
                              caption: Optional[str] = None,
                              label_map: Optional[Dict[str, int]] = None):
    """Generate a sequence example."""
    if FLAGS.video_root_path:
        video_path = os.path.join(FLAGS.video_root_path, video_path)
    imgs_encoded = extract_frames(video_path, start, end)

    # Initiate the sequence example.
    seq_example = tf.train.SequenceExample()

    # Add the label list as text and indices.
    if label_name:
        set_context_int("clip/label/index", label_map[label_name], seq_example)
        set_context_bytes("clip/label/text", label_name.encode(), seq_example)
    if caption:
        set_context_bytes("caption/string", caption.encode(), seq_example)
    # Add the frames as one feature per frame.
    for img_encoded in imgs_encoded:
        add_bytes_list("image/encoded", [img_encoded], seq_example)

    # Add spectrogram
    spectro = extract_spectro(video_path, start, end)
    add_spectrogram("melspec/feature/floats", spectro, seq_example)
    # Add audio.
    if FLAGS.decode_audio:
        audio = extract_audio(video_path, start, end)
        add_float_list("WAVEFORM/feature/floats", audio, seq_example)


    # Add other metadata.
    set_context_bytes("video/filename", video_path.encode(), seq_example)
    # Add start and time in micro seconds.
    set_context_int("clip/start/timestamp", int(1000000 * start), seq_example)
    set_context_int("clip/end/timestamp", int(1000000 * end), seq_example)
    return seq_example


def main(argv):
    del argv
    # reads the input csv.
    input_csv = pd.read_csv(FLAGS.csv_path, quotechar='"', skipinitialspace=True)
    if FLAGS.num_shards == -1:
        num_shards = int(math.sqrt(len(input_csv)))
    else:
        num_shards = FLAGS.num_shards
    # Set up the TFRecordWriters.
    basename = os.path.splitext(os.path.basename(FLAGS.csv_path))[0]
    shard_names = [
        os.path.join(FLAGS.output_path, f"{basename}-{i:05d}-of-{num_shards:05d}")
        for i in range(num_shards)
    ]
    writers = [tf.io.TFRecordWriter(shard_name) for shard_name in shard_names]

    if "label" in input_csv:
        unique_labels = list(set(input_csv["label"].values))
        l_map = {unique_labels[i]: i for i in range(len(unique_labels))}
    else:
        l_map = None

    if FLAGS.shuffle_csv:
        input_csv = input_csv.sample(frac=1)

    total_files = len(input_csv)
    with _close_on_exit(writers) as writers:
        for i in range(total_files):
            print(
                "Processing example %d of %d   (%d%%) \r" %
                (i+1, total_files, (i+1) * 100 / total_files),
                end="" if i < total_files - 1 else "\n")
            v = input_csv["video_path"].values[i]
            s = input_csv["start"].values[i]
            e = input_csv["end"].values[i]
            l = input_csv["label"].values[i] if "label" in input_csv else None
            c = input_csv["caption"].values[i] if "caption" in input_csv else None
            seq_ex = generate_sequence_example(
                v, s, e, label_name=l, caption=c, label_map=l_map)
            writers[i % len(writers)].write(seq_ex.SerializeToString())


if __name__ == "__main__":
#    import matplotlib.pyplot as plt
#    import librosa.display
#
#    S = extract_spectro("../test/--aE2O5G5WE.mp4", 0, 10)
#    fig, ax = plt.subplots()
#    S_dB = librosa.power_to_db(S, ref=np.max)
#    img = librosa.display.specshow(S_dB, x_axis='time',
#                                   y_axis='mel', sr=16000,
#                                   fmax=8000, ax=ax)
#    fig.colorbar(img, ax=ax, format='%+2.0f dB')
#    ax.set(title='Mel-frequency spectrogram')
#    plt.show()
#    exit()
    app.run(main)
