"""
This module contains functions for spawning processes to watch the Twitch stream programmatically.
"""

import os
import queue as q
import subprocess
import threading


def get_stream_url():
    ok = subprocess.run(
        ['streamlink', '--stream-url', 'https://www.twitch.tv/fftbattleground', 'best'],
        capture_output=True)
    return str(ok.stdout, encoding='utf-8')


def download_stream(queue: q.Queue, stop: threading.Event, fps=None):
    """Start watching twitch, this function blocks forever until `stop` is set or an error occurs in ffmpeg. It writes
    each frame as raw bytes into the queue.
    """
    try:
        if fps is None:
            fps = int(os.environ['FPS'])
        stream_url = get_stream_url()

        with subprocess.Popen(['ffmpeg', '-loglevel', 'panic', '-i', stream_url, '-filter:v',
                               'crop=990:740:145:260', '-q:v', '2', '-r', str(fps), '-f',
                               'mpjpeg', 'pipe:1'],
                              stdout=subprocess.PIPE,
                              bufsize=1024 * 1024 * 2) as proc:
            length = 0
            while not stop.is_set():
                line = proc.stdout.readline().strip()
                if line.startswith(b'--'):
                    continue
                if line.startswith(b'Content-type:'):
                    assert line.endswith(b'image/jpeg')
                    continue
                if line.startswith(b'Content-length:'):
                    length = int(line[len(b'Content-length: '):])
                    continue
                if line == b'':
                    queue.put(proc.stdout.read(length))
    finally:
        stop.set()
