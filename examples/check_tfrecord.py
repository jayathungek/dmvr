import sys
import json

from google.protobuf.json_format import MessageToJson
import tensorflow as tf


def inspect(dataset):
    print("inspecting")
    for d in dataset.take(1):
        ex = tf.train.Example()
        ex.ParseFromString(d.numpy())
        m = json.loads(MessageToJson(ex))
        print(m['features']['feature'].keys())


if __name__ == '__main__':
    path = sys.argv[1]
    raw_dataset = tf.data.TFRecordDataset(path)

    inspect(raw_dataset)
