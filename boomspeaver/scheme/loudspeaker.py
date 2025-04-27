# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boomspeaver.tools.data import get_repo_dir, get_value_from_dict, load_json_file


@dataclass
class LargeSignalParameters:
    xmax: float  # mm
    vd: float  # cm³
    P: float  # W
    Pmax: float  # W
    Ppeak: float  # W

    def __init__(self, data: dict[str, Any]):
        self.xmax = get_value_from_dict(
            data, "large_signal_parameters", "maximum_linear_excursion", "xmax"
        )
        self.vd = get_value_from_dict(
            data, "large_signal_parameters", "displaced_air_volume_at_xmax", "VD"
        )
        self.P = get_value_from_dict(
            data, "large_signal_parameters", "power_handling", "P"
        )
        self.Pmax = get_value_from_dict(
            data, "large_signal_parameters", "power_handling", "Pmax"
        )
        self.Ppeak = get_value_from_dict(
            data, "large_signal_parameters", "power_handling", "Ppeak"
        )


@dataclass
class QualityFactors:
    QMS: float  # mechanical quality factor
    QES: float  # electronic quality factor
    QTS: float  # total quality factor

    def __init__(self, data: dict[str, Any]):
        self.QMS = get_value_from_dict(
            data, "thiele_and_small", "quality_factors", "QMS"
        )
        self.QES = get_value_from_dict(
            data, "thiele_and_small", "quality_factors", "QES"
        )
        self.QTS = get_value_from_dict(
            data, "thiele_and_small", "quality_factors", "QTS"
        )


@dataclass
class ThieleSmallParameters:
    fS: float  # Hz
    quality_factors: QualityFactors

    def __init__(self, data: dict[str, Any]):
        self.fS = get_value_from_dict(
            data, "thiele_and_small", "resonance_frequency", "fS"
        )
        self.quality_factors = QualityFactors(data)


@dataclass
class VoiceCoil:
    Z: float  # Ω
    RE: float  # Ω
    LE: float  # mH
    VC_diameter: float  # mm
    HVC: float  # mm
    winding_material: str
    former_material: str

    def __init__(self, data: dict[str, Any]):
        self.Z = get_value_from_dict(data, "voice_coil", "nominal_impedance", "Z")
        self.RE = get_value_from_dict(data, "voice_coil", "DC_resistance", "RE")
        self.LE = get_value_from_dict(data, "voice_coil", "inductance", "LE")
        self.VC_diameter = get_value_from_dict(
            data, "voice_coil", "VC_diameter", "diameter"
        )
        self.HVC = get_value_from_dict(data, "voice_coil", "winding_height", "HVC")
        self.winding_material = get_value_from_dict(
            data, "voice_coil", "winding_material"
        )
        self.former_material = get_value_from_dict(
            data, "voice_coil", "former_material"
        )


@dataclass
class Magnet:
    Bl: float  # N/A
    Bl_sqrtRE: float  # N/√W
    HAG: float  # mm

    def __init__(self, data: dict[str, Any]):
        self.Bl = get_value_from_dict(data, "magnet", "force_factor", "Bl")
        self.Bl_sqrtRE = get_value_from_dict(
            data, "magnet", "motor_constant", "Bl_sqrtRE"
        )
        self.HAG = get_value_from_dict(data, "magnet", "air_gap_height", "HAG")


@dataclass
class Diaphragm:
    diameter: float  # mm
    SD: float  # cm²

    def __init__(self, data: dict[str, Any]):
        self.diameter = get_value_from_dict(
            data, "diaphragm", "effective_diameter", "diameter"
        )
        self.SD = get_value_from_dict(data, "diaphragm", "effective_area", "SD")


@dataclass
class MovingMass:
    MMS: float  # kg
    MMD: float  # kg

    def __init__(self, data: dict[str, Any]):
        self.MMS = get_value_from_dict(data, "moving_mass", "MMS")
        self.MMD = get_value_from_dict(data, "moving_mass_without_air_load", "MMD")


@dataclass
class Loudspeaker:
    def __init__(self, data: dict[str, Any]):
        self.sensitivity = get_value_from_dict(data, "sensitivity", "SPL1W")
        self.frequency_response = get_value_from_dict(data, "frequency_response")
        self.thiele_small = ThieleSmallParameters(data)
        self.large_signal_parameters = LargeSignalParameters(data)
        self.voice_coil = VoiceCoil(data)
        self.magnet = Magnet(data)
        self.diaphragm = Diaphragm(data)
        self.moving_mass = MovingMass(data)

    @classmethod
    def from_json(cls, input_path: Path) -> "Loudspeaker":
        """Load loudspeaker parameters from json file."""
        assert isinstance(input_path, Path)
        return cls(load_json_file(input_path))

    def print_main_params(self) -> None:
        """Show main loudspeaker parameters."""
        print(f"Sensitivity: {self.sensitivity} dB")
        print(f"Resonance Frequency (fS): {self.thiele_small.fS} Hz")
        print(f"QTS: {self.thiele_small.quality_factors.QTS}")
        print(f"Xmax: {self.large_signal_parameters.xmax} mm")
        print(f"Bl: {self.magnet.Bl} N/A")
        print(f"Moving Mass (MMS): {self.moving_mass.MMS} kg")


if __name__ == "__main__":

    repo_dir = get_repo_dir()
    input_config_path = repo_dir / "examples/prv_audio_6MB400_8ohm.json"
    loudspeaker = Loudspeaker.from_json(input_path=input_config_path)
    loudspeaker.print_main_params()
