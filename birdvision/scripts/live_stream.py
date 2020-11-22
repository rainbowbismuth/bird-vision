"""
This pygame application watches the stream live, displaying what it is reading off of each frame.
"""

import os
import sys
import threading
import time
from queue import Queue, Empty

import cv2
import numpy as np
import pygame

import birdvision.quiet
import birdvision.stream as stream_viewer
import birdvision.stream_state as stream_state
from birdvision.config import configure
from birdvision.constants import STREAM_WIDTH, STREAM_HEIGHT
from birdvision.node import Node
from birdvision.watcher import Watcher
from birdvision.object import ObjectModel


def add_reading_rects(image, finder_rect, rects):
    cv2.rectangle(image, finder_rect.top_left, finder_rect.bottom_right, (0, 255, 0), 1)
    for rect in rects:
        cv2.rectangle(image, rect.top_left, rect.bottom_right, (0, 0, 255), 1)


def main():
    configure()
    birdvision.quiet.silence_tensorflow()
    fps = int(os.environ['FPS'])

    stop_event = threading.Event()

    queue = Queue(maxsize=fps * 30)
    ffmpeg_thread = threading.Thread(
        target=lambda: stream_viewer.download_stream(queue, stop_event),
        daemon=True)
    ffmpeg_thread.start()

    pygame.init()
    pygame.font.init()
    font = pygame.font.Font('data/pygame/RobotoCondensed-Regular.ttf', 20)
    pygame.display.set_caption("Birb Brains Vision")
    pygame.display.set_icon(pygame.image.load('data/pygame/icon.png'))
    size = width, height = STREAM_WIDTH, STREAM_HEIGHT + 200
    screen = pygame.display.set_mode(size)
    black = 0, 0, 0

    surface = pygame.Surface((STREAM_WIDTH, STREAM_HEIGHT))

    # offsets = [(5, i * 28 + 5 + STREAM_HEIGHT) for i in range(6)] \
    #           + [(505, i * 28 + 5 + STREAM_HEIGHT) for i in range(6)]

    clock = pygame.time.Clock()
    saved_screens = 0
    watcher = Watcher()
    # object_model = ObjectModel()
    last_state = None

    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        f_start = time.monotonic()

        try:
            image = queue.get(block=False)
        except Empty:
            clock.tick(fps)
            continue

        jpeg_buf = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(jpeg_buf, flags=cv2.IMREAD_COLOR)
        if image is None:
            continue

        frame = Node(image)
        color_mapped = cv2.applyColorMap(frame.gray.image, cv2.COLORMAP_BONE)

        frame_info = watcher(frame)
        if frame_info != last_state and frame_info.state != stream_state.BLACK:
            print(frame_info)
            last_state = frame_info

        color_mapped = color_mapped[..., ::-1].copy()
        arr = pygame.surfarray.map_array(surface, color_mapped).swapaxes(0, 1)
        pygame.surfarray.blit_array(surface, arr)
        screen.blit(surface, surface.get_rect())

        # if stream_state.in_game(frame_info.state):
        #     objects = object_model(frame)
        #     for obj in objects:
        #         if obj.kind == 'None':
        #             continue
        #         kind = font.render(obj.kind, True, (100, 255, 100))
        #         screen.blit(kind, obj.rect.top_left)

        f_duration = time.monotonic() - f_start
        status_line = f'{queue.qsize():03d} {saved_screens:05d} {f_duration * 1000:.2f}ms'
        status_surf = font.render(status_line, True, (100, 255, 100))
        screen.blit(status_surf, (width - 200, 25))

        pygame.display.flip()
        screen.fill(black)

        clock.tick(fps)


if __name__ == '__main__':
    main()
