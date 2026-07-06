from ..common.robot_storage import RerunnableRobot
from ..bin.rerun import Arguments

from .types import Config, Individual


def main(args: Arguments):
    record = RerunnableRobot.load(args.robot_archive)
    print(record.misc)

    assert (genome := record.misc.get("genotype")) is not None
    assert (data := record.misc.get("genotype_rendering").get("data")) is not None

    ind = Individual(genome)
    ind._develop(data)
    print(ind)


if __name__ == "__main__":
    exit(main(Arguments.parse_command_line_arguments("Rerun evolved champions")))
