import argparse
import shutil
import time

from PyQt6.QtWidgets import QApplication, QDialog

from aapets.common import canonical_bodies
from aapets.common.canonical_bodies import get_all
from aapets.watchmaker.body_picker import BodyPicker
from aapets.watchmaker.config import WatchmakerConfig
from aapets.watchmaker.watchmaker import Watchmaker
from aapets.watchmaker.window import MainWindow


def main(args):
    if args.data_folder.exists() and any(args.data_folder.glob("*")):
        if not args.overwrite:
            raise RuntimeError(f"Output folder already exists, is not empty and overwriting was not requested")
        shutil.rmtree(args.data_folder)
    args.data_folder.mkdir(parents=True, exist_ok=True)

    if args.seed is None:
        args.seed = int(time.time())

    if not args.cache_folder.exists():
        args.cache_folder.mkdir(parents=True)

    app = QApplication([])

    if args.body is None:
        picker = BodyPicker(args)
        if picker.exec() == QDialog.DialogCode.Accepted:
            args.body = picker.get_body()
        else:
            args.body = next(iter(get_all().keys()))

    window = MainWindow(args)
    watchmaker = Watchmaker(window, args)

    watchmaker.reset()
    window.show()

    app.exec()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Main evolution script")
    WatchmakerConfig.populate_argparser(parser)
    # parser.print_help()
    parsed_config = parser.parse_args(namespace=WatchmakerConfig())
    parsed_config.update()

    main(parsed_config)
