import argparse
import boto3
import base64
from botocore.exceptions import ClientError


def parse_args():
    parser = argparse.ArgumentParser(description='Utility script to remove line at end of secrets.')
    parser.add_argument('--filename',
                        type=str,
                        help='Name of file to remove last line from.')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    filename = args.filename

    with open(filename, 'r') as f:
        file_contents = f.read()

    file_contents = '\n'.join(file_contents.split('\n'))

    with open(filename, 'w') as f:
        f.write(file_contents)