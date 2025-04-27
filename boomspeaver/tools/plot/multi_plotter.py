from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt

from boomspeaver.tools.plot.configs import Axis, Line, Plotter, Points, Stem, Subplot


class MultiPlotter:
    def __init__(self, config: Optional[Plotter] = None):
        self.subplots: List[Subplot] = []
        self.config = config if config else Plotter()
        self.figure = None

    def add_subplot(self, config: Subplot):
        config.validate()
        self.subplots.append(config)

    def _plot_element(self, ax, element, zorder: int):
        """Plot an element with a reversed zorder (first element gets highest zorder)."""
        if isinstance(element, Stem):
            markerline, stemline, baseline = ax.stem(
                element.x, element.y, basefmt=" ", label=element.label
            )
            markerline.set_zorder(zorder)
            stemline.set_zorder(zorder)
        elif isinstance(element, Points):
            scatter = ax.scatter(
                element.x,
                element.y,
                color=element.color,
                marker=element.marker,
                label=element.label,
            )
            scatter.set_zorder(zorder)
        elif isinstance(element, Line):
            (line_obj,) = ax.plot(
                element.x, element.y, color=element.color, label=element.label
            )
            line_obj.set_zorder(zorder)

    def compile_all(self):
        if not self.subplots:
            raise ValueError("No subplots defined.")

        fig, axes = plt.subplots(
            len(self.subplots),
            1,
            figsize=self.config.figsize,
            sharex=self.config.sharex,
            sharey=self.config.sharey,
        )
        if len(self.subplots) == 1:
            axes = [axes]

        for ax, config in zip(axes, self.subplots):
            axis_config = config.axis_config

            # Reverse the zorder assignment
            num_elements = len(config.chart_elements)
            for index, element in enumerate(config.chart_elements):
                zorder = num_elements - index  # Highest zorder for the first element
                self._plot_element(ax, element, zorder)

            ax.set_title(axis_config.title)
            ax.set_xlabel(axis_config.xlabel)
            ax.set_ylabel(axis_config.ylabel)

            if axis_config.logx:
                ax.set_xscale("log")
            if axis_config.logy:
                ax.set_yscale("log")

            if self.config.grid:
                ax.grid(True)
            ax.legend()

        if self.config.tight_layout:
            plt.tight_layout()
        self.figure = fig

    def plot(self):
        if not self.figure:
            self.compile_all()
        plt.show()

    def save(self, filepath: Path):
        assert isinstance(filepath, Path)
        assert not filepath.exists()

        if not self.figure:
            self.compile_all()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(filepath)
        print(f"Plot saved to {filepath}")


# Example usage:
if __name__ == "__main__":
    plotter_config = Plotter(
        figsize=(10, 10), grid=True, tight_layout=True, sharex=False, sharey=False
    )
    plotter = MultiPlotter(config=plotter_config)

    plot1 = Subplot(
        axis_config=Axis(
            title="Line Type",
            xlabel="Time [s]",
            ylabel="Force [N]",
        ),
        chart_elements=[Line([1, 2, 3], [6, 5, 4], label="Force 1", color="r")],
    )
    plot2 = Subplot(
        axis_config=Axis(
            title="Multi Line Type",
            xlabel="Time [s]",
            ylabel="Force [N]",
        ),
        chart_elements=[
            Line([1, 2, 3], [4, 5, 6], label="Force 1"),
            Line([1, 2, 3], [2, 3, 4], label="Force 2", color="g"),
        ],
    )
    plot3 = Subplot(
        axis_config=Axis(
            title="Stem Type",
            xlabel="Frequency [Hz]",
            ylabel="Amplitude [N]",
        ),
        chart_elements=[Stem([1, 2, 3], [6, 5, 4], label="Amplitude 1")],
    )
    plot4 = Subplot(
        axis_config=Axis(
            "Stem with Points with proper hierarchy", "Frequency [Hz]", "Amplitude [N]"
        ),
        chart_elements=[
            Points(
                [1, 2], [5, 3], marker="o", color="r", label="Peak Values"
            ),  # hierarchy most important on top
            Stem([1, 2, 3], [5, 3, 2], label="Amplitude 1"),
        ],
    )
    plot5 = Subplot(
        axis_config=Axis("Mixed type", "Time [s]", "Force [N]"),
        chart_elements=[
            Stem([1, 2, 3], [4, 5, 6], label="F(t)"),
            Line([1, 2, 3], [2, 3, 4], label="Line", color="g"),
            Points(
                [1, 2, 3],
                [4, 5, 6],
                color="r",
                marker="x",
                label="Peaks with wrong hierarchy",
            ),
        ],
    )

    # Add subplots to the plotter
    plotter.add_subplot(plot1)
    plotter.add_subplot(plot2)
    plotter.add_subplot(plot3)
    plotter.add_subplot(plot4)
    plotter.add_subplot(plot5)

    plotter.save(Path(__file__).resolve().parent / "output/plot_test.png")
    plotter.plot()
