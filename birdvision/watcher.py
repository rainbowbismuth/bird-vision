from dataclasses import dataclass
from typing import Optional

from birdvision import stream_state
from birdvision.character import CharacterModel
from birdvision.character.finder import String, StringFinder, light_text, dark_text
from birdvision.node import Node
from birdvision.rectangle import Rectangle
from birdvision.stream_state import StreamStateModel


@dataclass(frozen=True, eq=True)
class UnitVitals:
    curHP: Optional[int]
    maxHP: Optional[int]
    curMP: Optional[int]
    maxMP: Optional[int]
    curCT: Optional[int]


@dataclass(frozen=True, eq=True)
class UnitName:
    name: str
    job: str
    brave: Optional[int]
    faith: Optional[int]


@dataclass(frozen=True, eq=True)
class FrameInfo:
    state: str
    vitals: Optional[UnitVitals] = None
    name: Optional[UnitName] = None
    ability: Optional[str] = None


def string_to_int(s: String) -> Optional[int]:
    s = s.to_str()
    if len(s) == 0:
        return None
    return int(s)


class UnitVitalsReader:
    def __init__(self, character_model: CharacterModel):
        small_digit = character_model.read_small_digits
        self.curHP = StringFinder('curHP', Rectangle(350, 588, 60, 27), prepare_fn=light_text, reader_fn=small_digit)
        self.maxHP = StringFinder('maxHP', Rectangle(423, 601, 60, 27), prepare_fn=light_text, reader_fn=small_digit)
        self.curMP = StringFinder('curMP', Rectangle(350, 623, 60, 27), prepare_fn=light_text, reader_fn=small_digit)
        self.maxMP = StringFinder('maxMP', Rectangle(423, 636, 60, 27), prepare_fn=light_text, reader_fn=small_digit)
        self.curCT = StringFinder('curCT', Rectangle(350, 658, 60, 27), prepare_fn=light_text, reader_fn=small_digit)

    def __call__(self, frame: Node) -> UnitVitals:
        curHP = string_to_int(self.curHP(frame))
        maxHP = string_to_int(self.maxHP(frame))
        curMP = string_to_int(self.curMP(frame))
        maxMP = string_to_int(self.maxMP(frame))
        curCT = string_to_int(self.curCT(frame))
        return UnitVitals(curHP, maxHP, curMP, maxMP, curCT)


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
        name = self.name(frame).to_str()
        job = self.job(frame).to_str()
        brave = string_to_int(self.brave(frame))
        faith = string_to_int(self.faith(frame))
        return UnitName(name, job, brave, faith)


class Watcher:
    def __init__(self):
        self.stream_state_model = StreamStateModel()
        self.character_model = CharacterModel()
        self.left_unit_vitals = UnitVitalsReader(self.character_model)
        self.right_unit_name = UnitNameReader(self.character_model)
        self.ability_reader = StringFinder('ability', Rectangle(270, 122, 425, 58), prepare_fn=dark_text,
                                           reader_fn=self.character_model.read_alpha_num)

    def __call__(self, frame: Node) -> FrameInfo:
        state = self.stream_state_model(frame).name
        if not stream_state.in_game(state):
            return FrameInfo(state)

        if state == stream_state.GAME_SELECT_FULL:
            vitals = self.left_unit_vitals(frame)
            name = self.right_unit_name(frame)
            return FrameInfo(state, vitals=vitals, name=name)

        elif state == stream_state.GAME_SELECT_HALF_LEFT:
            vitals = self.left_unit_vitals(frame)
            return FrameInfo(state, vitals=vitals)

        elif state == stream_state.GAME_ABILITY_TAG:
            ability = self.ability_reader(frame).to_str()
            return FrameInfo(state, ability=ability)

        else:
            return FrameInfo(state)
