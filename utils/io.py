import os
import pickle


def all_file(dirname):
    fl = []
    for root, dirs, files in os.walk(dirname):
        for item in files:
            path = os.path.join(root, item)
            fl.append(path)
    return fl


def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return [line[:-1] for line in f]


def write_file(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    try:
        os.makedirs(dirname, exist_ok=True)
    except:
        pass
    with open(filename, 'w', encoding='utf-8') as f:
        for line in obj:
            f.write(str(line) + '\n')


def write_pkl(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def seg_array(inp, num=1):
    length = len(inp) // num + 1
    return [inp[ids * length:(ids + 1) * length] for ids in range(num)]


def read_dialog(filename):
    raw = read_file(filename)
    data = [[]]
    for line in raw:
        if line == '':
            data.append([])
        else:
            data[-1].append(line)
    data = [item for item in data if item != []]
    return data
