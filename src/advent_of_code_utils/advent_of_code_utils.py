"""
useful funcs for advent of code such as loading in the inputs
"""

import contextlib
import pathlib
import shutil
from dataclasses import dataclass
from math import ceil, log10
from pathlib import Path
from typing import Any, Callable

from IPython.display import Markdown, display
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_from_file(
    file: pathlib.Path,
    cast_as: type | list[type] = str, delimiter: str = ','
) -> list:
    """
    returns the contents of a file as a list where each item in the list was
    delimited in the file by the character passed.

    Each line is then cast as a the types specified.
    """
    with open(file, 'r') as f:
        line_list = [line.strip() for line in f.readlines()]

    parsed_lines = []
    if isinstance(cast_as, list):
        for line in line_list:
            parsed_lines.append(
                [cast(item) for cast, item in
                    zip(cast_as, line.split(delimiter))]
            )
    else:
        for line in line_list:
            parsed_lines.append(cast_as(line))

    return parsed_lines


@dataclass
class ParseConfig:
    """
    Class to contain a delimiter and how to parse each split item.
    """
    delimiter: str
    parser: 'Callable | ParseConfig | list[Callable | ParseConfig]'


def parse_from_file(
    file: pathlib.Path, parse_config: ParseConfig,
    unnest_single_items: bool = False
) -> list:
    """
    parses an entire file using the parse config

    the file is passed in it's entirity to parse_string() so a file may need
    to be split by \n before being parsed line by line.

    can be specified to unnest single items if parsing only appends one item
    """
    with open(file) as f:
        entire_file = f.read().rstrip()
    return parse_string(entire_file, parse_config, unnest_single_items)


def parse_string(
    string: str, parse_config: ParseConfig,  unnest_singles: bool = False
) -> list:
    """
    recursively parses a string using the parse_config to return a multi-level
    list object.

    The parse_config.parser argument:
    - parse_config.delimiter will be used to split the string into segments
    - parse_config.delimiter can be an empty string. This will create a list of
    every character in a string.
    - If the parser specified is Callable, this function will be called on each
    delimited string segment.
    - If the parser is a ParseConfig, this parse config will be applied to
    to each delimited string segment, creating sub-lists.
    - If the parser is a list the Callable or ParseConfig items in the list
    will be used to parse the corresponding delimited string segment by index.

    if specified lists of a single item will return that item instead of a list
    """
    # break up string using delimiter
    if parse_config.delimiter == '':
        segments = [char for char in string]
    else:
        segments = string.split(parse_config.delimiter)

    # parse string segments
    parsed_output = []
    if isinstance(parse_config.parser, ParseConfig):
        parsed_output = [
            parse_string(segment, parse_config.parser, unnest_singles)
            for segment in segments
        ]
    elif isinstance(parse_config.parser, list):
        if len(parse_config.parser) < len(segments):
            raise ValueError(
                'ParseConfig list not as long as delimited segments: '
                f'{parse_config}, {segments}')
        temp = []
        for parser, segment in zip(parse_config.parser, segments):
            if parser is None:
                continue
            elif isinstance(parser, ParseConfig):
                temp.append(parse_string(segment, parser, unnest_singles))
            else:
                temp.append(parser(segment))
        parsed_output = temp
    else:
        parsed_output = [parse_config.parser(segment) for segment in segments]

    # unnest output if specified to do so
    if unnest_singles and len(parsed_output) == 1:
        return parsed_output.pop()
    else:
        return parsed_output


def markdown(*lines: str) -> None:
    """Shortcut for using IPython.display.display to render markdown"""
    display(Markdown('\n'.join(lines)))


# let's make some gifs
def create_gif_from_images(
        image_paths: list, gif_name: Path, duration: int) -> None:
    """creates a gif with the images passed"""
    with contextlib.ExitStack() as stack:
        images = (
            stack.enter_context(Image.open(file))
            for file in sorted(image_paths)
        )
        image = next(images)
        image.save(
            fp=gif_name, format='GIF', append_images=images, save_all=True,
            duration=duration, loop=0)


def create_plot_images(
    frames: list,
    plot_generator: Callable[[Any, plt.Axes], None],
    title: str = None, append_iteration: bool = False
) -> Path:
    """
    creates a gif of each frame rendered by the plot generator

    the plot generator func must take one item from frames and an axis
    """

    # create temp directory for putting images into
    image_path = Path('img_temp')
    image_path.mkdir(exist_ok=True)

    n_iterations = len(frames)

    def i_str(value: int) -> str:
        """formats a string with the right number of leading 0s"""
        return f'{value:0{ceil(log10(n_iterations + 1))}d}'

    # create images and save them without displaying
    for iteration, frame in tqdm(
        enumerate(frames), desc='generating frames', total=n_iterations
    ):

        fig, ax = plt.subplots()
        plot_generator(frame, ax)

        title_segments = []
        if title is not None:
            title_segments.append(title)
        if append_iteration:
            title_segments.append(f'iteration: {i_str(iteration)}')

        if len(title_segments) > 0:
            ax.set_title(' - '.join(title_segments))

        fig.savefig(image_path / f'temp_{i_str(iteration)}.png')
        plt.close(fig)

    return image_path


def create_gif(
    frames: list,
    plot_generator: Callable[[Any, plt.Axes], None], filename: Path,
    frame_duration_ms: int = 50, title: str = None,
    append_iteration: bool = False
) -> None:
    """top level gif creation function"""
    filename = Path(filename).with_suffix('.gif')
    # create images
    image_path = create_plot_images(
        frames, plot_generator, title=title, append_iteration=append_iteration)
    # create a gif from them
    create_gif_from_images(
        image_path.glob('*.png'), filename, frame_duration_ms)
    # clean up temp image directory
    shutil.rmtree(image_path)


def plot_grid(
    grid: list[list[int]], ax: plt.Axes, remove_ticks: bool = True
) -> None:
    """Adds a pcolourmesh of a 2D grid of integers to the axis passed"""
    ax.pcolormesh(np.flipud(grid))
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
