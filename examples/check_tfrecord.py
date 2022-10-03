import sys
import json

from google.protobuf.json_format import MessageToJson
import tensorflow as tf
from tensorflow import python as tf_python


def inspect(dataset):
    for example in dataset:
        result = tf.train.SequenceExample.FromString(example)
        print(result)

if __name__ == '__main__':
    path = sys.argv[1]
    raw_dataset = tf_python.python_io.tf_record_iterator(path)

    inspect(raw_dataset)
