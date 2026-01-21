import argparse
import os
import time
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QDialog
from rich.progress import Progress

from aapets.watchmaker.consent import ConsentDialog
from aapets.watchmaker.plotting import Plotter
from aapets.watchmaker.types import HillClimberSelector, RandomSelector, Selector
from ...common.canonical_bodies import get_all
from ..body_picker import BodyPicker
from ..config import WatchmakerConfig, RunTypes
from ..watchmaker import Watchmaker
from ..window import WatchmakerWindow


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

    if args.seed is None:
        args.seed = int(time.time())

    args.run_id = time.strftime(f"%Y%m%d-%H%M%S-{args.seed}")
    args.data_folder = args.data_folder.joinpath(args.run_type.value).joinpath(args.run_id)

    if args.plot_from is not None:
        if not args.plot_from.exists():
            raise RuntimeError(f"Plotting data under folder '{args.plot_from}' was requested but this folder does not exist")
        else:
            Plotter.do_final_plots(args.where(data_folder=args.plot_from))
            return None

    if args.data_folder.exists() and any(args.data_folder.glob("*")):
        if not args.overwrite:
            raise RuntimeError(f"Output folder already exists, is not empty and overwriting was not requested")
        for item in args.data_folder.glob("*"):
            os.remove(item)
            print("rm", item)
    args.data_folder.mkdir(parents=True, exist_ok=True)

    if not args.cache_folder.exists():
        args.cache_folder.mkdir(parents=True)

    if args.symlink:
        symlink = args.data_folder.parent.joinpath("last")
        symlink.unlink(missing_ok=True)
        symlink.symlink_to(args.data_folder.name, target_is_directory=True)

    if args.run_type is RunTypes.HUMAN:
        return run_interactive(args)

    else:
        assert args.max_evaluations is not None, "Automated mode requires a target number of evaluations"

        return run_automated(args)


def run_interactive(args):
    app = QApplication([])

    if not args.skip_consent:
        consent = ConsentDialog(args.run_id)
        if consent.exec() != QDialog.DialogCode.Accepted:
            exit(1)

    if args.body is None:
        picker = BodyPicker(args.cache_folder)
        if picker.exec() == QDialog.DialogCode.Accepted:
            args.body = picker.get_body()
        else:
            args.body = next(iter(get_all().keys()))
        parsed_config.update()

    window = WatchmakerWindow(args)
    watchmaker = Watchmaker(window=window, config=args)

    watchmaker.reset()
    window.show()

    app.exec()

    if args.plots:
        Plotter.do_final_plots(args)


def run_automated(args):
    selector = {
        RunTypes.HILL: HillClimberSelector(),
        RunTypes.RANDOM: RandomSelector(args.seed)
    }[args.run_type]
    watchmaker = Watchmaker(config=args)
    watchmaker.reset()

    mute = (args.verbosity <= 0)
    with Progress(disable=mute and False) as progress:
        task = progress.add_task("[green]Computing...", total=args.max_evaluations + 1)

        run = True
        while run:
            selection_ix, _ = selector.select(watchmaker.population)
            run = watchmaker.next_generation(selection_ix)

            progress.update(task, completed=watchmaker.evaluations)

        champion = watchmaker.re_evaluate_champion(gif=True)
        progress.update(task, advance=1)

    if args.plots:
        Plotter.do_final_plots(args)

    if not mute:
        print("Champion:", champion, f"fitness={champion.fitness}")
    return champion.fitness


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Main evolution script")
    WatchmakerConfig.populate_argparser(parser)
    # parser.print_help()
    parsed_config = parser.parse_args(namespace=WatchmakerConfig())
    parsed_config.update()

    main(parsed_config)
