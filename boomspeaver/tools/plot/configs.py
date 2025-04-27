from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class Points:
    x: List[float]
    y: List[float]
    color: Optional[str] = None
    marker: Optional[str] = "o"
    label: Optional[str] = None


@dataclass
class Line:
    x: List[float]
    y: List[float]
    color: Optional[str] = None
    label: Optional[str] = None


@dataclass
class Stem:
    x: List[float]
    y: List[float]
    label: Optional[str] = None


@dataclass
class Axis:
    title: str
    xlabel: str
    ylabel: str
    logx: bool = False
    logy: bool = False


@dataclass
class Subplot:
    axis_config: Axis
    chart_elements: List[Union[Line, Stem, Points]]

    def validate(self):
        if not self.chart_elements:
            raise ValueError("chart_elements cannot be empty.")


@dataclass
class Plotter:
    figsize: Tuple[int, int] = (10, 10)
    grid: bool = True
    tight_layout: bool = True
    sharex: bool = False
    sharey: bool = False
