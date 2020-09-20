"""Console script for datasets."""
import argparse
import sys
import os
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from datasets.utils import DataSchema
from datasets import TFDataset, TextlineParser


def convert2tfrecords(dataset_file_path, file_suffix, tfrecord_file_path, n_workers):
    filenames = os.listdir(dataset_file_path)
    if file_suffix is None:
        filenames = [f for f in list(filenames)]
    else:
        filenames = [f for f in list(filenames) if f.endswith(file_suffix)]

    def processor(filepath, filename, tfrecord_filename):
        token_dicts = None
        data_field_list = []
        data_field_list.append(DataSchema(name='query', processor='to_np', type=tf.int32,
                                          dtype='int32', shape=(None,), is_with_len=True))
        label_field = DataSchema(name='label', processor='to_np',
                                 type=tf.float32, dtype='float32', shape=(1,), is_with_len=False)
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path=filepath, file_suffix=filename)
        generator.to_tfrecords(tfrecord_filename)
        return tfrecord_filename

    task_param_list = [tuple(dataset_file_path, filename, tfrecord_file_path + '/' + str(i) + '.tfrecord')
                       for filename, i in zip(filenames, len(filenames))]
    pool = ThreadPoolExecutor(max_workers=n_workers)
    for result in pool.map(processor, task_param_list):
        print(result, 'finish')


def main():
    """Console script for datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "datasets.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
