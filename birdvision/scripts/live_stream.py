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

import birdvision.character as character
import birdvision.quiet
import birdvision.stream_state as stream_state
import birdvision.stream_viewer as stream_viewer
from birdvision.config import configure
from birdvision.node import Node


def add_reading_rects(image, finder_rect, rects):
    cv2.rectangle(image, finder_rect.top_left, finder_rect.bottom_right, (0, 255, 0), 1)
    for rect in rects:
        cv2.rectangle(image, rect.top_left, rect.bottom_right, (0, 0, 255), 1)


def main():
    configure()
    birdvision.quiet.silence_tensorflow()
    fps = int(os.environ['FPS'])

    stop_event = threading.Event()

    queue = Queue(maxsize=fps * 10)
    ffmpeg_thread = threading.Thread(
        target=lambda: stream_viewer.download_stream(queue, stop_event),
        daemon=True)
    ffmpeg_thread.start()

    pygame.init()
    pygame.font.init()
    font = pygame.font.Font('data/pygame/RobotoCondensed-Regular.ttf', 20)
    pygame.display.set_caption("Birb Brains Vision")
    pygame.display.set_icon(pygame.image.load('data/pygame/icon.png'))
    size = width, height = 990, 740 + 200
    screen = pygame.display.set_mode(size)
    black = 0, 0, 0

    char_model = character.CharacterModel()
    stream_state_model = stream_state.StreamStateModel()
    finders = character.finders_from_model(char_model)
    finders.append(stream_state.StreamStateFinder(stream_state_model))

    surface = pygame.Surface((990, 740))

    offsets = [(5, i * 28 + 5 + 740) for i in range(6)] + [(505, i * 28 + 5 + 740) for i in range(6)]
    finder_names = [font.render(finder.name, True, (255, 255, 255)) for finder in finders]

    clock = pygame.time.Clock()
    letter_count = 0

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

        # TODO: Work on the combined piece, i.e. don't read ability text unless AbilityTag, etc
        for i, finder in enumerate(finders):
            readings = list(finder.find(frame))

            if not readings:
                continue

            # TODO: Acting rects again..
            # rects = [reading.notes["absolute_rect"] for reading in readings if reading.notes]
            # add_reading_rects(color_mapped, finder.rect, rects)

            if isinstance(finder, character.CharacterFinder):
                value = character.found_to_string(readings)
            else:
                value = readings[0].value

            # for j, (prob, char) in enumerate(reading.prob_chars):
            #     if prob < 0.51:
            #         cv2.imwrite(
            #             f'/Volumes/RAM_Disk_512MB/letters/{reading.name}_{char}_{letter_count}.png',
            #             reading.images[j])
            #         letter_count += 1

            certainty = np.product([reading.certainty for reading in readings])

            offset = offset_x, offset_y = offsets[i]
            screen.blit(finder_names[i], offset)
            text_surf = font.render(value, True, (255, 255, 255))
            screen.blit(text_surf, (offset_x + 100, offset_y))
            certainty_surf = font.render(f'{certainty:.2f}', True, (255, 255, 255))
            screen.blit(certainty_surf, (offset_x + 400, offset_y))

        color_mapped = color_mapped[..., ::-1].copy()
        arr = pygame.surfarray.map_array(surface, color_mapped).swapaxes(0, 1)
        pygame.surfarray.blit_array(surface, arr)
        screen.blit(surface, surface.get_rect())
        pygame.display.flip()
        screen.fill(black)

        f_duration = time.monotonic() - f_start
        print(f'{queue.qsize():03d}', f'{letter_count:05d}', f'{f_duration * 1000:.2f}ms')

        clock.tick(fps)


if __name__ == '__main__':
    main()
