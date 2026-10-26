"""Replay all nine serial checkpoints, including held-out prestretch predictions."""
import sys
from train_9_sbar_serial import main as train_main


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else list(arguments)
    train_main([*arguments, '--predict-only'])


if __name__ == '__main__':
    main()
