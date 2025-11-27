import argparse
import os
import time

from PyQt6.QtWidgets import QApplication, QDialog

from aapets.watchmaker.consent import ConsentDialog
from ...common.canonical_bodies import get_all
from ..body_picker import BodyPicker
from ..config import WatchmakerConfig
from ..watchmaker import Watchmaker
from ..window import MainWindow


# TODO:
# - Slider for temperature (mutation deviation)

def main(args):
    # app = QApplication([])
    # movie = QMovie("tmp/watchmaker/test-run/0_0.gif")
    # label = QLabel()
    # label.setMovie(movie)
    # movie.start()
    # label.setMinimumSize(label.sizeHint().expandedTo(QSize(100, 100)))
    # label.show()
    # app.exec()
    # exit(42)

    # model, data = compile_world(make_world(get("spider45").spec, None, camera_angle=0))
    # mujoco_viewer.launch(model, data)
    # # single_frame_renderer(
    # #     model, data, width=960, height=960,
    # #     camera="apet1_tracking-cam",
    # #     save=True, save_path="tmp/watchmaker/test-run/__test.png"
    # # )
    # exit(42)

    if args.data_folder.exists() and any(args.data_folder.glob("*")):
        if not args.overwrite:
            raise RuntimeError(f"Output folder already exists, is not empty and overwriting was not requested")
        for item in args.data_folder.glob("*"):
            os.remove(item)
            print("rm", item)
    args.data_folder.mkdir(parents=True, exist_ok=True)

    if args.seed is None:
        args.seed = int(time.time())

    if not args.cache_folder.exists():
        args.cache_folder.mkdir(parents=True)

    app = QApplication([])

    if not args.skip_consent:
        consent = ConsentDialog(args.data_folder)
        if consent.exec() != QDialog.DialogCode.Accepted:
            exit(1)

    if args.body is None:
        picker = BodyPicker(args)
        if picker.exec() == QDialog.DialogCode.Accepted:
            args.body = picker.get_body()
        else:
            args.body = next(iter(get_all().keys()))
        parsed_config.update()

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
