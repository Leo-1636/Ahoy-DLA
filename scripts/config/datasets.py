from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from config.settings import DatasetConfig


class OmniDLA(BaseModel):
    name: ClassVar[str] = "omnidla/OmniDLA"
    path: ClassVar[Path] = DatasetConfig.path / "OmniDLA"
    splits: ClassVar[list[str]] = ["train", "val", "test"]
    class_names: ClassVar[dict[int, str]] = {
        0: "None",
        1: "Caption",
        2: "Footnote",
        3: "Formula",
        4: "List-item",
        5: "Page-footer",
        6: "Page-header",
        7: "Picture",
        8: "Section-header",
        9: "Table",
        10: "Text",
        11: "Title",
    }
    
class DocLayNet(BaseModel):
    name: ClassVar[str] = "ds4sd/DocLayNet-v1.2"
    path: ClassVar[Path] = DatasetConfig.path / "DocLayNet"
    splits: ClassVar[list[str]] = ["train", "val", "test"]
    class_names: ClassVar[dict[int, str]] = {
        0: "None",
        1: "Caption",
        2: "Footnote",
        3: "Formula",
        4: "List-item",
        5: "Page-footer",
        6: "Page-header",
        7: "Picture",
        8: "Section-header",
        9: "Table",
        10: "Text",
        11: "Title",
    }
    label_map: ClassVar[dict[int, int]] = {
        0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
    }

class D4LA(BaseModel):
    name: ClassVar[str] = "iic/D4LA"
    path: ClassVar[Path] = DatasetConfig.path / "D4LA"
    splits: ClassVar[list[str]] = ["train", "test"]
    class_names: ClassVar[dict[int, str]] = {
        1: "DocTitle",
        2: "ParaTitle",
        3: "ParaText",
        4: "ListText",
        5: "RegionTitle",
        6: "Date",
        7: "LetterHead",
        8: "LetterDear",
        9: "LetterSign",
        10: "Question",
        11: "OtherText",
        12: "RegionKV",
        13: "RegionList",
        14: "Abstract",
        15: "Author",
        16: "TableName",
        17: "Table",
        18: "Figure",
        19: "FigureName",
        20: "Equation",
        21: "Reference",
        22: "Footer",
        23: "PageHeader",
        24: "PageFooter",
        25: "Number",
        26: "Catalog",
        27: "PageNumber",
    }

class DocSynth300K(BaseModel):
    name: ClassVar[str] = "juliozhao/DocSynth300K"
    path: ClassVar[Path] = DatasetConfig.path / "DocSynth300K"
    splits: ClassVar[list[str]] = ["train"]
    class_names: ClassVar[dict[int, str]] = {
        0: "QR code",
        1: "advertisement",
        2: "algorithm",
        3: "answer",
        4: "author",
        5: "barcode",
        6: "bill",
        7: "blank",
        8: "bracket",
        9: "breakout",
        10: "byline",
        11: "caption",
        12: "catalogue",
        13: "chapter title",
        14: "code",
        15: "correction",
        16: "credit",
        17: "dateline",
        18: "drop cap",
        19: "editor's note",
        20: "endnote",
        21: "examinee information",
        22: "fifth-level title",
        23: "figure",
        24: "first-level question number",
        25: "first-level title",
        26: "flag",
        27: "folio",
        28: "footer",
        29: "footnote",
        30: "formula",
        31: "fourth-level section title",
        32: "fourth-level title",
        33: "header",
        34: "headline",
        35: "index",
        36: "inside",
        37: "institute",
        38: "jump line",
        39: "kicker",
        40: "lead",
        41: "marginal note",
        42: "matching",
        43: "mugshot",
        44: "option",
        45: "ordered list",
        46: "other question number",
        47: "page number",
        48: "paragraph",
        49: "part",
        50: "play",
        51: "poem",
        52: "reference",
        53: "sealing line",
        54: "second-level question number",
        55: "second-level title",
        56: "section",
        57: "section title",
        58: "sidebar",
        59: "sub section title",
        60: "subhead",
        61: "subsub section title",
        62: "supplementary note",
        63: "table",
        64: "table caption",
        65: "table note",
        66: "teasers",
        67: "third-level question number",
        68: "third-level title",
        69: "title",
        70: "translator",
        71: "underscore",
        72: "unordered list",
        73: "weather forecast",
    }
    label_map: ClassVar[dict[int, int]] = {
        7: -1, 8: -1, 53: -1, 71: -1,
        11: 0, 64: 0,
        20: 1, 29: 1, 65: 1,
        30: 2,
        24: 3, 44: 3, 45: 3, 46: 3, 54: 3, 67: 3, 72: 3,
        27: 4, 28: 4, 47: 4,
        33: 5,
        0: 6, 1: 6, 5: 6, 23: 6, 43: 6,
        22: 7, 25: 7, 31: 7, 32: 7, 55: 7, 57: 7, 59: 7, 60: 7, 61: 7, 68: 7,
        6: 8, 63: 8,
        2: 9, 3: 9, 4: 9, 9: 9, 10: 9, 12: 9, 14: 9, 15: 9, 16: 9, 17: 9, 18: 9, 19: 9, 21: 9, 26: 9, 35: 9, 36: 9, 37: 9, 38: 9, 39: 9, 40: 9, 41: 9, 42: 9, 48: 9, 49: 9, 50: 9, 51: 9, 52: 9, 56: 9, 58: 9, 62: 9, 66: 9, 70: 9, 73: 9,
        13: 10, 34: 10, 69: 10
    }

class DocGenome(BaseModel):
    name: ClassVar[str] = "U4R/DocGenome"
    path: ClassVar[Path] = DatasetConfig.path / "DocGenome"
    splits: ClassVar[list[str]] = []
    class_names: ClassVar[dict[int, str]] = {
        0: "Algorithm",
        1: "Caption",
        2: "Equation",
        3: "Figure",
        4: "Footnote",
        5: "List",
        6: "Table",
        7: "Text",
        8: "Text-EQ",
        9: "Title",
        10: "PaperTitle",
        11: "Code",
        12: "Abstract",
    }