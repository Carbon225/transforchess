from typing import List
from pathlib import Path
from time import sleep
from shutil import rmtree
import argparse


def get_old_checkpoints(dir: Path) -> List[Path]:
    checkpoints = [d for d in dir.glob('checkpoint-*') if d.is_dir()]
    checkpoints.sort(key=lambda x: int(x.name.removeprefix('checkpoint-')))
    return checkpoints[:-4]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    try:
        while True:
            for checkpoint in get_old_checkpoints(args.path):
                rmtree(checkpoint)
            sleep(5)
    except KeyboardInterrupt:
        pass
