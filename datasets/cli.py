"""Console script for datasets."""
import argparse
import sys
import os
import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datasets.utils import DataSchema
from datasets import TFDataset, TextlineParser
from functools import wraps

# color print ref : https://blog.csdn.net/qq_34857250/article/details/79673698


def loop_input(a_func):
    @wraps(a_func)
    def decorated(*args, **kwargs):
        while True:
            result, msg = a_func(*args, **kwargs)
            if result is not None:
                return result
            else:
                print('\033[1;31m' + msg + '\033[0m')
    return decorated


@loop_input
def filepath_input(msg):
    filepath = input('请输入' + msg + '的文件路径:')
    filepath = filepath.strip()
    if os.path.exists(filepath):
        return filepath, None
    else:
        return None, filepath + 'is not find'


@loop_input
def shape_input():
    line = input('请输入数据的shape[e.g. 15,3]:')
    try:
        shape = [float(x) for x in line.split(',')]
        return shape, None
    except ValueError:
        return None, '输入格式错误，请输入各个维度，按照,隔开 e.g. 15,3'


@loop_input
def choose_input(choose_list, type_text):
    print(' '.join(choose_list))
    line = input(' 请从以上选项选出' + type_text + ':')
    if line in choose_list:
        return line, None
    else:
        return None, '请从选项当中选择，可以copy，请勿超出选项范围'


@loop_input
def name_input(msg):
    line = input(msg)
    line = line.strip()
    if len(line) > 0:
        return line, None
    else:
        return None, '请输入非空字符'


@loop_input
def yesno_input(msg, default='y'):
    if default == 'y':
        tail = '[Y/n]:'
        result = True
    else:
        tail = '[y/N]:'
        result = False
    line = input(msg + tail)
    line = line.strip()
    if line == '':
        return result
    elif line in {'yes', 'YES', 'Yes', 'y', 'Y'}:
        return True
    elif line in {'no', 'NO', 'No', 'n', 'N'}:
        return False
    else:
        return None, '请填入如下选项: y,n'

# def choose_processor_input():


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


class CodeGenerator(object):
    def __init__(self):
        self.schema_list = []
        self.add_info = []
        self.label_len = 1
        self.feat_flags = []
        self.label_names = []
        self.feature_names = []
        self.init_code = 'batch_size num_epochs is_shuffle file_path file_suffix'
        self.code = """
        token_dicts = TokenDicts('tests/data/dicts', {'query': 0})
        data_field_list = []
        data_field_list.append(DataSchema(name='query', processor='to_tokenid', type=tf.int32,
                                          dtype='int32', shape=(None,), is_with_len=True, token_dict_name='query'))
        data_field_list.append(DataSchema(
            name='width', processor='to_np', type=tf.float32, dtype='float32', shape=(4)))
        label_field = DataSchema(
            name='label', processor='to_np', type=tf.float32, dtype='float32', shape=(1,))
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path=file_path, file_suffix=file_suffix)
        dataset = generator.generate_dataset(
            batch_size=batch_size, num_epochs=num_epochs, is_shuffle=is_shuffle)
        for _ in enumerate(dataset):
            pass
        """
        self.convert_dicts = {
            'wav': ['mfcc2', 'audio', 'float', [None, 13]],
            'pcm': ['mfcc2', 'audio', 'float', [None, 13]],
            'mp3': ['mfcc2', 'audio', 'float', [None, 13]],
            'jpg': ['img_read', 'img', 'float', None],
            'png': ['img_read', 'img', 'float', None],
            'JPG': ['img_read', 'img', 'float', None]
        }

    def _get_field_processor(self, colum_list):
        is_same_shape = True
        processor, type, dtype, shape = self._data_field_check(colum_list[0])
        for data, index in zip(colum_list, range(len(colum_list))):
            types = self._data_field_check(data)
            if processor != types[0]:
                print(processor, ' not match line', index, colum_list[index], types[0])
                sys.exit(-1)
            if type != types[1]:
                print(type, ' not match line', index, colum_list[index], types[1])
                sys.exit(-1)
            if dtype != types[2]:
                print(dtype, ' not match line', index, colum_list[index], types[2])
                sys.exit(-1)
            if shape != types[3]:
                is_same_shape = False
        if shape is None:
            shape = shape_input()
        if processor in {'to_np', 'to_tokenid'} and not is_same_shape:
            shape = [None]
        print('processor', processor, 'type', type, 'dtype', dtype, 'shape', shape, 'for data colums')
        print(colum_list[:3])
        return [processor, type, dtype, shape]

    def _write_token_dicts(self, f, label_schema, features_schema):
        dict_names = set()
        for name, schema in zip(self.label_names, label_schema):
            if schema[0] in {'to_tokenid'}:
                dict_name = name_input('input dict name for' + name + ' :')
                dict_names.add(dict_name)
        for name, schema in zip(self.feature_names, features_schema):
            if schema[0] in {'to_tokenid'}:
                dict_name = name_input('input dict name for feature :' + name + ' :')
                dict_names.add(dict_name)
        if len(dict_names) == 0:
            f.write('token_dicts = None\n')
        else:
            name_list = '{' + ','.join(["'%s':0" % (x) for x in dict_names]) + '}'
            f.write('token_dicts = TokenDicts(\'dicts\', ' + name_list + ')\n')

    def _write_schema_list(self, f, schema_list, name_list, list_name):
        f.write(list_name + ' = []\n')
        # data_field_list.append(DataSchema(
        #    name='width', processor='to_np', type=tf.float32, dtype='float32', shape=(4)))
        for name, schema in zip(name_list, schema_list):
            shape_text = ','.join([x if x else 'None' for x in schema[3]])
            f.write("%s.append(DataSchema(name='%s', processor='%s', type='%s', shape=(%s)))\n" % (list_name, name, schema[0], schema[1], shape_text))

    def generate_code(self):
        workspace = name_input('输入代码生成目录 : ')
        if not os.path.exists(workspace):
            os.mkdir(workspace)
        code_file = workspace + '/dataset_reader.py'
        data_file = filepath_input('输入数据文件')
        file_suffix = input('输入数据文件后缀 : ')
        file_suffix = file_suffix.strip()
        #code_file = filepath_input('写入代码文件')
        lines = []
        with open(data_file, 'r') as f:
            for line in f:
                lines.append(line.strip('\n'))
                if len(lines) > 20:
                    break
        with open(code_file, 'w') as f:
            f.write('\n'.join(['import tensorflow as tf', 'from datasets import TextlineParser',
                               'from datasets import TFDataset', 'from datasets.utils import TokenDicts, DataSchema']))
            f.write('\n')
            f.write("file_path = " + data_file + '\n')
            if file_suffix == '':
                f.write("file_suffix = None\n")
            else:
                f.write("file_suffix = '" + file_suffix + "'\n")
            f.write("batch_size = 64\n")
            f.write("num_epochs = 1\n")
            f.write("is_shuffle = True\n")

            self._parser_header(lines[0])
            label_items = []
            feature_items = []
            print('lines', lines)
            for line in lines[1:]:
                items = line.split('\t')
                items = items[len(self.add_info):]
                label_items.append(items[0:self.label_len])
                feature_items.append(items[self.label_len:])
            label_items = np.asarray(label_items)
            feature_items = np.asarray(feature_items)
            print('items shape', label_items.shape)
            label_schema = []
            features_schema = []
            for colum in range(label_items.shape[1]):
                colum_list = label_items[:, colum].tolist()
                label_schema.append(self._get_field_processor(colum_list))
            for colum in range(feature_items.shape[1]):
                colum_list = label_items[:, colum].tolist()
                features_schema.append(self._get_field_processor(colum_list))
            print('add_info', self.add_info)
            print('label schema:', label_schema)
            print('feature schema:', features_schema)
            self._write_token_dicts(f, label_schema, features_schema)
            self._write_schema_list(f, label_schema, self.label_names, 'label_schema_list')
            self._write_schema_list(f, features_schema, self.feature_names, 'feature_schema_list')
            f.write("parser = TextlineParser(token_dicts, feature_schema_list, label_schema_list)\n")
            f.write("generator = TFDataset(parser=parser, file_path=file_path, file_suffix=file_suffix)\n")
            f.write("dataset = generator.generate_dataset(batch_size=batch_size, num_epochs=num_epochs, is_shuffle=is_shuffle)\n")
            f.write("for _ in enumerate(dataset):\n    pass\n")

    def _parser_header(self, head):
        items = head.split('\t')
        self.label_len = 0
        self.feat_flags = []
        i = 0
        while i < len(items):
            if items[i].startswith('label'):
                break
            else:
                self.add_info.append(items[i])
            i += 1
        while i < len(items):
            if items[i].startswith('label'):
                self.label_len += 1
                self.label_names.append(items[i])
            else:
                break
            i += 1
        while i < len(items):
            self.feature_names.append(items[i])
            if items[i].startswith('fe_'):
                self.feat_flags.append(True)
            else:
                self.feat_flags.append(False)
            i += 1

    def _data_field_check(self, text):
        file_suffix = text.split('.')[-1]
        if file_suffix in self.convert_dicts:
            return self.convert_dicts[file_suffix]
        items = text.split(' ')
        is_float = True
        is_int = True
        for item in items:
            if not item.isnumeric():
                is_int = False
            try:
                float(item)
            except ValueError:
                is_float = False
        if is_int:
            return ['to_np', 'class', 'int', len(items)]
        if is_float:
            return ['to_np', 'num', 'float', len(items)]
        return ['to_tokenid', 'text', 'int', len(items)]


def main():
    """Console script for datasets."""
    #parser = argparse.ArgumentParser()
    #parser.add_argument('_', nargs='*')
    #args = parser.parse_args()
    code_generator = CodeGenerator()
    code_generator.generate_code()
    #print("Arguments: " + str(args._))
    # print("Replace this message by putting your code into "
    #      "datasets.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
