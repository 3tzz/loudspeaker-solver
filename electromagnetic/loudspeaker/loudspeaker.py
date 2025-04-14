import json
from pathlib import Path


class LargeSignalParameters:
    def __init__(self, data):
        self.xmax = data.get("maximum_linear_excursion", {}).get("xmax", 0)  # mm
        self.vd = data.get("displaced_air_volume_at_xmax", {}).get("VD", 0)  # cm³
        power_handling = data.get("power_handling", {})
        self.P = power_handling.get("P", 0)  # W
        self.Pmax = power_handling.get("Pmax", 0)  # W
        self.Ppeak = power_handling.get("Ppeak", 0)  # W


class QualityFactors:
    def __init__(self, data):
        self.QMS = data.get("QMS", 0)
        self.QES = data.get("QES", 0)
        self.QTS = data.get("QTS", 0)


class ThieleSmallParameters:
    def __init__(self, data):
        self.fS = data.get("resonance_frequency", {}).get("fS", 0)  # Hz
        self.quality_factors = QualityFactors(data.get("quality_factors", {}))


class VoiceCoil:
    def __init__(self, data):
        self.Z = data.get("nominal_impedance", {}).get("Z", 0)  # Ω
        self.RE = data.get("DC_resistance", {}).get("RE", 0)  # Ω
        self.LE = data.get("inductance", {}).get("LE", 0)  # mH
        self.VC_diameter = data.get("VC_diameter", {}).get("diameter", 0)  # mm
        self.HVC = data.get("winding_height", {}).get("HVC", 0)  # mm
        self.winding_material = data.get("winding_material", "")
        self.former_material = data.get("former_material", "")


class Magnet:
    def __init__(self, data):
        self.Bl = data.get("force_factor", {}).get("Bl", 0)  # N/A
        self.Bl_sqrtRE = data.get("motor_constant", {}).get("Bl_sqrtRE", 0)  # N/√W
        self.HAG = data.get("air_gap_height", {}).get("HAG", 0)  # mm


class Diaphragm:
    def __init__(self, data):
        self.diameter = data.get("effective_diameter", {}).get("diameter", 0)  # mm
        self.SD = data.get("effective_area", {}).get("SD", 0)  # cm²


class MovingMass:
    def __init__(self, data):
        self.MMS = data.get("MMS", 0)  # g
        self.MMD = data.get("MMD", 0)  # g


class Loudspeaker:
    def __init__(self, data):
        self.sensitivity = data.get("sensitivity", {}).get("SPL1W", 0)  # dB
        self.frequency_response = data.get("frequency_response", {})
        self.thiele_small = ThieleSmallParameters(data.get("thiele_and_small", {}))
        self.large_signal_parameters = LargeSignalParameters(
            data.get("large_signal_parameters", {})
        )
        self.voice_coil = VoiceCoil(data.get("voice_coil", {}))
        self.magnet = Magnet(data.get("magnet", {}))
        self.diaphragm = Diaphragm(data.get("diaphragm", {}))
        self.moving_mass = MovingMass(data)


def load_loudspeaker_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return Loudspeaker(data)


def main():
    # Automatically get the path using pathlib
    file_path = Path(__file__).parent / "../example/prv_audio_6MB400_8ohm.json"

    loudspeaker = load_loudspeaker_data(file_path)

    print(f"Sensitivity: {loudspeaker.sensitivity} dB")
    print(f"Resonance Frequency (fS): {loudspeaker.thiele_small.fS} Hz")
    print(f"QMS: {loudspeaker.thiele_small.quality_factors.QMS}")
    print(f"QES: {loudspeaker.thiele_small.quality_factors.QES}")
    print(f"Xmax: {loudspeaker.large_signal_parameters.xmax} mm")
    print(f"Bl: {loudspeaker.magnet.Bl} N/A")
    print(f"Moving Mass (MMS): {loudspeaker.moving_mass.MMS} g")


if __name__ == "__main__":
    main()
