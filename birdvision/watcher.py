import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2

from birdvision import stream_state
from birdvision.character import CharacterModel
from birdvision.character.finder import String, StringFinder, light_text, dark_text
from birdvision.node import Node
from birdvision.object import ObjectModel
from birdvision.rectangle import Rectangle
from birdvision.stream_state import StreamStateModel
from birdvision.stream_state.model import StreamState


@dataclass(frozen=True, eq=True)
class UnitVitals:
    curHP: Optional[int]
    maxHP: Optional[int]
    curMP: Optional[int]
    maxMP: Optional[int]
    curCT: Optional[int]


@dataclass(frozen=True, eq=True)
class UnitName:
    name: Optional[str]
    job: Optional[str]
    brave: Optional[int]
    faith: Optional[int]


@dataclass(frozen=True, eq=True)
class FrameInfo:
    state: str
    vitals: Optional[UnitVitals] = None
    name: Optional[UnitName] = None
    ability: Optional[str] = None


LOW_CERTAINTY_CUT_OFF = 0.5


def write_low_certainty_node(path: str, node: Node):
    Path(path).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f'{path}/{uuid4()}.png', node.image)


def record_low_certainty_string(tag: str, s: String):
    low_certainty_path = os.environ.get('RECORD_LOW_CERTAINTY')
    for i, confidence in enumerate(s.confidences):
        if confidence > LOW_CERTAINTY_CUT_OFF:
            continue

        if low_certainty_path is not None:
            path = f'{low_certainty_path}/{tag}'
            write_low_certainty_node(path, s.nodes[i])


def record_low_certainty_stream_state(s: StreamState, frame: Node):
    low_certainty_path = os.environ.get('RECORD_LOW_CERTAINTY')
    if s.certainty <= LOW_CERTAINTY_CUT_OFF and low_certainty_path is not None:
        path = f'{low_certainty_path}/stream_state'
        write_low_certainty_node(path, frame)


class UnitVitalsReader:
    def __init__(self, character_model: CharacterModel):
        small_digit = character_model.read_small_digits
        self.curHP = StringFinder('curHP', Rectangle(350, 588, 60, 27), prepare_fn=light_text, reader_fn=small_digit)
        self.maxHP = StringFinder('maxHP', Rectangle(423, 601, 60, 27), prepare_fn=light_text, reader_fn=small_digit)
        self.curMP = StringFinder('curMP', Rectangle(350, 623, 60, 27), prepare_fn=light_text, reader_fn=small_digit)
        self.maxMP = StringFinder('maxMP', Rectangle(423, 636, 60, 27), prepare_fn=light_text, reader_fn=small_digit)
        self.curCT = StringFinder('curCT', Rectangle(350, 658, 60, 27), prepare_fn=light_text, reader_fn=small_digit)

    def __call__(self, frame: Node) -> UnitVitals:
        curHP = self.curHP(frame)
        maxHP = self.maxHP(frame)
        curMP = self.curMP(frame)
        maxMP = self.maxMP(frame)
        curCT = self.curCT(frame)
        record_low_certainty_string('curHP', curHP)
        record_low_certainty_string('maxHP', maxHP)
        record_low_certainty_string('curMP', curMP)
        record_low_certainty_string('maxMP', maxMP)
        record_low_certainty_string('curCT', curCT)
        return UnitVitals(curHP.to_int(), maxHP.to_int(), curMP.to_int(), maxMP.to_int(), curCT.to_int())


class UnitNameReader:
    def __init__(self, character_model: CharacterModel):
        small_digit = character_model.read_small_digits
        alpha_num = character_model.read_alpha_num
        self.name = StringFinder('name', Rectangle(610, 545, 320, 40), prepare_fn=dark_text, reader_fn=alpha_num,
                                 find_spaces=True)
        self.job = StringFinder('job', Rectangle(610, 595, 320, 40), prepare_fn=dark_text, reader_fn=alpha_num,
                                find_spaces=True)
        self.brave = StringFinder('brave', Rectangle(725, 653, 42, 30), prepare_fn=dark_text, reader_fn=small_digit)
        self.faith = StringFinder('faith', Rectangle(877, 653, 42, 30), prepare_fn=dark_text, reader_fn=small_digit)

    def __call__(self, frame: Node) -> UnitName:
        name = self.name(frame)
        job = self.job(frame)
        brave = self.brave(frame)
        faith = self.faith(frame)
        record_low_certainty_string('name', name)
        record_low_certainty_string('job', job)
        record_low_certainty_string('brave', brave)
        record_low_certainty_string('faith', faith)
        return UnitName(name.to_str(), job.to_str(), brave.to_int(), faith.to_int())


class Watcher:
    def __init__(self):
        self.stream_state_model = StreamStateModel()
        self.character_model = CharacterModel()
        self.left_unit_vitals = UnitVitalsReader(self.character_model)
        self.right_unit_name = UnitNameReader(self.character_model)
        self.ability_reader = StringFinder('ability', Rectangle(270, 122, 425, 58), prepare_fn=dark_text,
                                           reader_fn=self.character_model.read_alpha_num, find_spaces=True)

    def __call__(self, frame: Node) -> FrameInfo:
        state = self.stream_state_model(frame)
        record_low_certainty_stream_state(state, frame)
        state_name = state.name
        if not stream_state.in_game(state_name):
            return FrameInfo(state_name)

        if state_name == stream_state.GAME_SELECT_FULL:
            vitals = self.left_unit_vitals(frame)
            name = self.right_unit_name(frame)
            return FrameInfo(state_name, vitals=vitals, name=name)

        elif state_name == stream_state.GAME_SELECT_HALF_LEFT:
            vitals = self.left_unit_vitals(frame)
            return FrameInfo(state_name, vitals=vitals)

        elif state_name == stream_state.GAME_ABILITY_TAG:
            ability = self.ability_reader(frame)
            record_low_certainty_string('ability', ability)
            return FrameInfo(state_name, ability=ability.to_str())

        else:
            return FrameInfo(state_name)
