from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from config.settings import DatasetConfig


class DocLayNet(BaseModel):
    name: ClassVar[str] = "ds4sd/DocLayNet-v1.2"
    path: ClassVar[Path] = DatasetConfig.path / "DocLayNet"
    splits: ClassVar[list[str]] = ["train", "validation", "test"]
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
