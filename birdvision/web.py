"""
A small local web application, to view the results of our automated testing.
"""
from uuid import UUID

import cv2
from dotenv import load_dotenv, find_dotenv
from flask import Flask, Markup, Response, render_template

import birdvision.quiet
import birdvision.testing
from birdvision.node import Node

load_dotenv(find_dotenv())
birdvision.quiet.silence_tensorflow()
TESTS = birdvision.testing.run_all_tests()

app = Flask(__name__, static_url_path='')

# To make it easier to develop. They parse very fast anyways.
app.config['TEMPLATES_AUTO_RELOAD'] = True


def to_png(image):
    return Response(cv2.imencode('.png', image)[1].tobytes(), mimetype='image/png')


@app.template_filter('node_img')
def node_image_filter(node: Node):
    if node is None:
        return ''
    return Markup(render_template('node_img.html', node=node))


@app.route('/')
def show_index():
    failures = list(TESTS.failures())
    return render_template('index.html', failures=failures)


@app.route('/test/<int:index>')
def show_test(index):
    result = TESTS.results[index]
    test_template = f'tests/{result.data.__class__.__name__}.html'
    test_render = Markup(render_template(test_template, data=result.data))
    return render_template('test.html', result=result, test_render=test_render)


@app.route('/test/<int:index>/frame')
def show_test_frame(index):
    result = TESTS.results[index]
    return to_png(result.frame.image)


@app.route('/node/<node_id>')
def show_test_node(node_id):
    node_id = UUID(node_id)
    node = TESTS.get_node(node_id)
    return render_template('node.html', node=node)


@app.route('/node/<node_id>/image')
def show_test_node_image(node_id):
    node_id = UUID(node_id)
    node = TESTS.get_node(node_id)
    return to_png(node.image)
