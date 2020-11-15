# bird-vision

This is a machine vision project to read information off of a live stream of Final Fantasy Tactics, called [FFTBG](https://www.twitch.tv/fftbattleground).

All of this code is written in Python, particulary with `opencv` and `tensorflow`. Everything you need should be in the Pipfile *except* for `ffmpeg` and `stream-link`, if you choose to run the code that uses them.

There is a lot of data that goes along with this project, and to download it you will need to make use of [Git Large File Storage](https://git-lfs.github.com/).


# Quick start

Once you have everything installed, and the data downloaded, you can do the following from the project's root directory.


Run tests:
```
python -m birdvision.scripts.run_tests
```

Train new models:
```
python -m birdvision.scripts.train_models
```

Or, if you want to run the web viewer, to visualize test cases:
```
FLASK_APP=birdvision.web python -m flask run
```