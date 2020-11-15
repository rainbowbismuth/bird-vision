"""
A small local web application, to view the stream live, and to view the results of our automated testing.
"""
from typing import List

from flask import Flask, Response, render_template, send_from_directory
from dotenv import load_dotenv, find_dotenv
import birdvision.testing

import birdvision.quiet
import cv2

from birdvision.rectangle import Rectangle

load_dotenv(find_dotenv())
birdvision.quiet.silence_tensorflow()
birdvision.testing.run_all_tests()

app = Flask(__name__, static_url_path='')

# To make it easier to develop. They parse very fast anyways.
app.config['TEMPLATES_AUTO_RELOAD'] = True


def to_png(image):
    return Response(cv2.imencode('.png', image)[1].tobytes(), mimetype='image/png')


def add_reading_rects_cropped(image, local_rects: List[Rectangle]):
    move_by = 200 // len(local_rects)
    for i, r in enumerate(local_rects):
        cv2.rectangle(image, r.top_left, r.bottom_right, (i * move_by, i * move_by, 255 - i * move_by), 1)


@app.route('/')
def show_index():
    failures = [result for result in birdvision.testing.RESULTS if not result.ok]
    return render_template('index.html', failures=failures)


@app.route('/test/<int:index>')
def show_test(index):
    result = birdvision.testing.RESULTS[index]
    return render_template('test.html', result=result)


@app.route('/test/<int:index>/char/<int:char_idx>')
def show_test_char(index, char_idx):
    result = birdvision.testing.RESULTS[index]
    image = result.notes["crops"][char_idx]
    return to_png(image)


@app.route('/test/<int:index>/prepared')
def show_test_prepared(index):
    result = birdvision.testing.RESULTS[index]
    image = result.notes["prepared"]
    return to_png(image)


@app.route('/test/<int:index>/prepared-with-rects')
def show_test_prepared_rects(index):
    result = birdvision.testing.RESULTS[index]
    image = result.notes["prepared"]
    color_mapped = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    add_reading_rects_cropped(color_mapped, result.notes["local_rects"])
    return to_png(color_mapped)


@app.route('/static/<path:path>')
def show_static(path):
    return send_from_directory('', path)
