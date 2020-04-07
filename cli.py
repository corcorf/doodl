"""Console script for doodl."""
import argparse
import sys


def get_arguments():
    """
    Get arguments for running a supermarket simulation and visualisation
    """
    description = 'Specify parameters for the supermarket simulation.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--n_checkouts", dest='n_checkouts', type=int, nargs='?', default=4,
        help='Number of checkouts to be simulated'
    )
    parser.add_argument(
        "--checkout_rate", dest='checkout_rate', type=float, nargs='?',
        default=0.5,
        help='Probability that a customer clears each checkout each minute'
    )
    parser.add_argument(
        '--save_records', dest='save_records', type=bool, nargs='?',
        default=False, const=True,
        help='If used simulated customer records will be saved to file'
    )
    parser.add_argument(
        '--filename', dest='filename', type=str, nargs='?',
        default="simulated_customer_records.csv",
        help='name of file that customer records that should be saved to'
    )
    parser.add_argument(
        '--date', dest='date', type=str, nargs='?', default=None,
        help='date of shpping day to be simulated'
    )
    parser.add_argument(
        '--show_colour', dest='show_colour', type=bool, nargs='?',
        default=False, const=True,
        help='If used visualisation includes colour representing entry time'
    )
    parser.add_argument(
        '--image_path', dest='image_path', type=str, nargs='?',
        default=None, const="images",
        help='if specified, images from each frame will be saved to file'
    )
    parser.add_argument(
        '--gif_path', dest='gif_path', type=str, nargs='?',
        default=None, const="images/doodl.gif",
        help='If used animation frames in image_path will be used to \
        create a gif at the specified file path'
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(get_arguments())  # pragma: no cover
