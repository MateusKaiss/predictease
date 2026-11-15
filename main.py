from predictease.cli import parse_args
from predictease.core import run

if __name__ == '__main__':
    run(**vars(parse_args()))
