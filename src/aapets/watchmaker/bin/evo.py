import argparse
import time

from PyQt6.QtWidgets import QApplication

from aapets.common import canonical_bodies
from aapets.watchmaker.body_picker import BodyPicker
from aapets.watchmaker.config import WatchmakerConfig
from aapets.watchmaker.watchmaker import Watchmaker
from aapets.watchmaker.window import MainWindow


def main(args):
    if args.seed is None:
        args.seed = int(time.time())

    BodyPicker.get_image(canonical_bodies.get("spider"))
    app = QApplication([])

    if args.body is None:
        print("Pick a body")
        args.body = BodyPicker().exec()
        print(">>", args.body)

    window = MainWindow(args)
    watchmaker = Watchmaker(window, args)

    print(args)

    window.show()

    app.exec()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Main evolution script")
    WatchmakerConfig.populate_argparser(parser)
    # parser.print_help()
    parsed_config = parser.parse_args(namespace=WatchmakerConfig())

    main(parsed_config)
