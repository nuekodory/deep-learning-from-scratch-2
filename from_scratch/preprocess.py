import sys
import os
from pathlib import Path
from natto import MeCab


if __name__ == '__main__':
    # macOS
    os.environ['MECAB_PATH'] = '/usr/local/lib/libmecab.dylib'
    os.environ['MECAB_CHARSET'] = 'utf-8'

    nm = MeCab()

    args = sys.argv
    input_path = Path(args[1])
    output_path = Path(args[2])

    with input_path.open() as f, output_path.open(mode='w') as wf:
        for line in f:
            nodes = nm.parse(line, as_nodes=True)
            surfaces = [node.surface for node in nodes]
            wf.write(' '.join(surfaces) + '\n')


