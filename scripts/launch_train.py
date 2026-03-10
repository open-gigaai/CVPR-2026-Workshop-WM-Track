import os
import sys
import argparse
import paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
    from giga_train import launch_from_config
    launch_from_config(config_path)


if __name__ == '__main__':
    main()
