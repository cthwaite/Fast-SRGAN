# coding: utf-8

import argparse
import logging
import os
from itertools import islice
from typing import Tuple

import cv2
import numpy as np
import tqdm

log = logging.getLogger("fast-srgan-video")


def decode_fourcc(fourcc_fmt: float) -> Tuple[str]:
    fmt = int(fourcc_fmt)
    return tuple(chr((fmt >> (8 * i)) & 0xFF) for i in range(4))


def iter_frames(low_res):
    while True:
        ret, low_frame = low_res.read()
        if low_frame is None:
            return
        yield low_frame


def write_one_frame(model, low_frame, out):
    """Upscale and write one frame into an output video.
    """
    low_frame = low_frame.astype(np.float32) / 255.0

    sup_frame, *_ = model.predict(np.expand_dims(low_frame, axis=0))
    height, width = sup_frame.shape[:2]

    sup_frame = ((sup_frame + 1) / 2.0) * 255
    sup_frame = sup_frame.astype(np.int8)

    out.write(sup_frame)


def write_one_image(model, low_frame, output_path: os.PathLike):
    """Upscale and write one frame into an image.
    """
    low_frame = low_frame.astype(np.float32) / 255.0

    sup_frame, *_ = model.predict(np.expand_dims(low_frame, axis=0))
    height, width = sup_frame.shape[:2]

    sup_frame = ((sup_frame + 1) / 2.0) * 255
    sup_frame = sup_frame.astype(np.int8)

    sup_frame = cv2.cvtColor(sup_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, sup_frame)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument(
        "--one-frame",
        action="store_true",
        help="Output one upscaled frame from the video as a PNG ",
    )
    parser.add_argument(
        "--one-second", action="store_true", help="Output one second of upscaled video"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset in frames for --one-frame, or seconds for --one-second",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Do not print verbose logs"
    )

    args = parser.parse_args()

    input_path = args.input

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level)

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    if args.output:
        output_path = args.output
    else:
        head, tail = os.path.split(input_path)
        name, ext = os.path.splitext(tail)
        output_path = os.path.join(head, f"{name} upscaled{ext}")

    log.info("Output will be saved to %s", output_path)

    log.info("Processing %s", input_path)

    # Read video
    low_res = cv2.VideoCapture(input_path)
    if low_res is None:
        log.error("Failed to open %s", input_path)
        return
    frame_count = int(low_res.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(low_res.get(cv2.CAP_PROP_FPS))
    if args.offset:
        if args.one_second and (args.offset * fps) >= frame_count:
            print(
                f"{args.offset} is not a valid time for this video (valid range [0, {frame_count/fps}])"
            )
            return
        elif args.one_frame and args.offset >= frame_count:
            print(
                f"{args.offset} is not a valid frame index for this video (valid range [0, {frame_count}])"
            )
            return
        else:
            log.warning(
                "--offset specified without --one-second or --one-frame, will have no effect"
            )
    width = int(low_res.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(low_res.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fmt = decode_fourcc(low_res.get(cv2.CAP_PROP_FOURCC))
    log.info("Source image: %d x %d, %d fps", width, height, fps)
    log.info("Source format is %s", "".join(fmt))
    SCALE_FACTOR = 4
    up_width, up_height = width * SCALE_FACTOR, height * SCALE_FACTOR
    log.info("Upscaling to %d x %d", up_width, up_height)
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (up_width, up_height)
    )

    log.info("Loading model...")

    from tensorflow import keras

    model = keras.models.load_model("models/generator.h5")
    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)

    frame_iter = iter_frames(low_res)
    if args.one_frame:
        offset_frame = args.offset
        frame_iter = enumerate(frame_iter)
        for index, frame in frame_iter:
            if index == offset_frame:
                write_one_image(model, frame, "out.png")
                break
    elif args.one_second:
        offset_begin = fps * args.offset
        offset_end = offset_begin + fps
        frame_iter = islice(frame_iter, offset_begin, offset_end)
        frame_iter = tqdm.tqdm(frame_iter, total=offset_end - offset_begin)
        for frame in frame_iter:
            write_one_frame(model, frame, out)
    else:
        frame_iter = tqdm.tqdm(frame_iter, total=frame_count)

        for frame in frame_iter:
            write_one_frame(model, frame, out)

    low_res.release()
    out.release()


if __name__ == "__main__":
    main()
