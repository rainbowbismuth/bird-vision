"""
This module contains functions for spawning processes to watch the Twitch stream programmatically.

While there are many references to a RAM Disk, using one is not necessary, just encouraged. This code works by having
ffmpeg repeatedly write images into a specific path, and having another python thread read and delete them. You don't
want any of this touching your hard-disk or SSD, and having a finite-size disk will help in-case something goes wrong
with ffmpeg or the code consuming those images. JPEGs are a lot bigger than individual frame in a compressed video file.
"""

import os
import queue as q
import subprocess
import threading
import time
from pathlib import Path

import cv2


def get_ram_disk_path(ram_disk_path=None) -> Path:
    if ram_disk_path is None:
        return Path(os.environ['RAM_DISK_PATH'])
    else:
        return Path(ram_disk_path)


def get_stream_url():
    ok = subprocess.run(['streamlink', '--stream-url', 'https://www.twitch.tv/fftbattleground', 'best'])
    return ok.stdout


def download_stream(stop: threading.Event, fps=None, ram_disk_path=None):
    """Use ffmpeg to start downloading the stream into JPGs on your RAM disk. This function blocks until ffmpeg
    is killed, or `stop` is set. If `stop` is set, then the ffmpeg process is killed if alive."""
    if fps is None:
        fps = int(os.environ['FPS'])

    ram_disk_path = get_ram_disk_path(ram_disk_path)
    stream_url = get_stream_url()
    process = subprocess.Popen(['ffmpeg', '-i', stream_url, '-filter:v', 'crop=990:740:145:260', '-qscale:v', '3', '-r',
                                str(fps), '-f', 'image2', f'{ram_disk_path}/output_%05d.jpg'])
    try:
        while process.poll() is None and not stop.is_set():
            time.sleep(1.0)
    finally:
        if process.poll() is None:
            process.kill()


def ram_disk_reader(queue: q.Queue, stop: threading.Event, ram_disk_path=None):
    """Loop until `stop` is set, reading images in `ram_disk_path`, deleting them and adding them to `queue`. When
    this code exits for any reason, `stop` is set."""
    ram_disk_path = get_ram_disk_path(ram_disk_path)

    try:
        while not stop.is_set():
            paths = sorted(ram_disk_path.glob('*.jpg'))
            for path in paths:
                try:
                    image = cv2.imread(path.as_posix())
                    path.unlink(missing_ok=True)
                    queue.put(image, block=False)
                except q.Full:
                    continue
    finally:
        stop.set()
