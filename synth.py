#!/usr/bin/env python3
"""
Simple multi-instrument synth + mixer template.

Save as synth.py and run:
    python3 synth.py

Dependencies:
    numpy

Outputs:
    - per-instrument WAVs in the current directory (e.g. melody.wav)
    - final_mix.wav (16-bit, mono)

Design goals:
 - readable, small, easy to extend
 - token lists similar to the Bash template (use "R" for rest)
 - supports waveforms: sine, square, triangle, sawtooth
 - per-note ADSR envelope with safe scaling when ADSR sum > duration
"""

from dataclasses import dataclass
import numpy as np
import wave
import struct
import math
import shutil
import os
from typing import List, Union

# -------------------- Global settings --------------------
SR = 44100               # sample rate
BIT_DEPTH = 16           # 16-bit output
BPM = 120                # tempo, used to compute default token length
BEAT_FRACTION = 0.5      # default token length = beat * BEAT_FRACTION
DEFAULT_GAIN_DB = 0.0
DEFAULT_TOKEN_DUR = (60.0 / BPM) * BEAT_FRACTION  # seconds per token

# Master output filename
FINAL_NAME = "final_mix.wav"

# -------------------- Note / frequency utilities --------------------
NOTE_OFFSETS = {
    'C': -9, 'C#': -8, 'Db': -8,
    'D': -7, 'D#': -6, 'Eb': -6,
    'E': -5,
    'F': -4, 'F#': -3, 'Gb': -3,
    'G': -2, 'G#': -1, 'Ab': -1,
    'A': 0, 'A#': 1, 'Bb': 1,
    'B': 2
}

def note_to_freq(token: str) -> float:
    """
    Convert token like 'E4' or 'C#5' to frequency in Hz.
    If token is numeric (e.g. '440' or '440.0') returns float.
    Raises ValueError for unknown tokens.
    """
    token = token.strip()
    if token == "R":
        return 0.0
    # numeric frequency?
    try:
        return float(token)
    except ValueError:
        pass

    # parse note name + octave
    # examples: C4, C#4, Db4
    # find trailing digits for octave
    i = len(token) - 1
    while i >= 0 and token[i].isdigit():
        i -= 1
    name = token[:i+1]
    octave_str = token[i+1:]
    if octave_str == "":
        raise ValueError(f"No octave in token '{token}'")
    octave = int(octave_str)
    # normalize name (allow b as flat)
    name = name.replace('♯', '#').replace('♭', 'b')
    if name not in NOTE_OFFSETS:
        raise ValueError(f"Unknown note name '{name}' in token '{token}'")
    semitone_offset = NOTE_OFFSETS[name] + (octave - 4) * 12
    freq = 440.0 * (2.0 ** (semitone_offset / 12.0))
    return freq

# -------------------- Waveform generators --------------------
def synth_waveform(wave_type: str, freq: float, dur: float, sr: int):
    """
    Return a numpy float32 array in range [-1,1] for the requested waveform.
    If freq == 0 (rest), returns zeros.
    """
    n = int(math.ceil(dur * sr))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / sr
    if freq == 0.0:
        return np.zeros(n, dtype=np.float32)
    phase = 2.0 * math.pi * freq * t
    if wave_type == "sine":
        return np.sin(phase).astype(np.float32)
    elif wave_type == "square":
        return np.sign(np.sin(phase)).astype(np.float32)
    elif wave_type == "triangle":
        # triangle from saw: 2 * abs(2*(t*freq - floor(t*freq + 0.5))) - 1
        saw = 2.0 * ( (t * freq) - np.floor(t * freq + 0.5) )
        tri = 2.0 * np.abs(saw) - 1.0
        return tri.astype(np.float32)
    elif wave_type == "sawtooth":
        saw = 2.0 * ( (t * freq) - np.floor(t * freq + 0.5) )
        return saw.astype(np.float32)
    else:
        raise ValueError(f"Unsupported wave type: {wave_type}")

# -------------------- ADSR envelope --------------------
def apply_adsr(signal: np.ndarray, sr: int, attack: float, decay: float, sustain_level: float, release: float):
    n = len(signal)
    dur = n / sr
    # scale ADSR proportionally if sum > dur*0.95
    max_ar = dur * 0.95
    if attack + decay + release > max_ar:
        scale = max_ar / (attack + decay + release)
        attack *= scale; decay *= scale; release *= scale
    # segments sizes in samples
    a_samples = int(round(attack * sr))
    d_samples = int(round(decay * sr))
    r_samples = int(round(release * sr))
    s_samples = max(0, n - (a_samples + d_samples + r_samples))
    # build envelope
    env = np.zeros(n, dtype=np.float32)
    idx = 0
    if a_samples > 0:
        env[idx:idx+a_samples] = np.linspace(0.0, 1.0, a_samples, endpoint=False)
        idx += a_samples
    if d_samples > 0:
        env[idx:idx+d_samples] = np.linspace(1.0, sustain_level, d_samples, endpoint=False)
        idx += d_samples
    if s_samples > 0:
        env[idx:idx+s_samples] = sustain_level
        idx += s_samples
    if r_samples > 0:
        env[idx:idx+r_samples] = np.linspace(sustain_level, 0.0, r_samples, endpoint=False)
        idx += r_samples
    # any remaining tail set to 0
    return signal * env

# -------------------- WAV write --------------------
def write_wav_mono(filename: str, samples: np.ndarray, sr: int, bit_depth: int = 16):
    """
    Write float32 samples in [-1,1] to a 16-bit PCM WAV file.
    """
    assert bit_depth == 16, "Only 16-bit implemented"
    # clip
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767.0).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())

# -------------------- Instrument dataclass --------------------
@dataclass
class Instrument:
    name: str
    wave: str = "sine"
    gain_db: float = -6.0
    octshift: int = 0
    tokens: List[str] = None
    normalize: bool = True
    adsr: dict = None  # {'attack':0.01,'decay':0.05,'sustain':0.8,'release':0.05}

# -------------------- Default instruments (mirrors bash template) --------------------
instruments = [
    Instrument(
        name="melody",
        wave="triangle",
        gain_db=-8.0,
        octshift=0,
        normalize=True,
        adsr={'attack':0.01,'decay':0.02,'sustain':0.9,'release':0.06},
        tokens="E7 E7 E7 R E7 E7 E7 R E7 G7 C7 D7 E7 R R R F7 F7 F7 F7 F7 E7 E7 E7 E7 D7 D7 E7 D7 R G7 R".split()
    ),
    Instrument(
        name="acc",
        wave="sawtooth",
        gain_db=-14.0,
        octshift=0,
        normalize=True,
        adsr={'attack':0.005,'decay':0.02,'sustain':0.85,'release':0.04},
        tokens="E4 E4 E4 R E4 E4 E4 R E4 G4 C4 D4 E4 R R R F4 F4 F4 F4 F4 E4 E4 E4 E4 D4 D4 E4 D4 R G4 R".split()
    ),
    Instrument(
        name="bass",
        wave="sine",
        gain_db=-6.0,
        octshift=-1,
        normalize=True,
        adsr={'attack':0.005,'decay':0.03,'sustain':0.95,'release':0.05},
        tokens="E2 R R R E2 R R R E2 R R R E2 R R R F2 R R R F2 R R R E2 R R R D2 R R R".split()
    ),
    Instrument(
        name="drum",
        wave="triangle",
        gain_db=-6.0,
        octshift=0,
        normalize=True,
        adsr={'attack':0.001,'decay':0.03,'sustain':0.3,'release':0.05},
        tokens="E3 R E3 R E3 R E3 R E3 R C3 R E3 R E3 R F3 R F3 R F3 R E3 R D3 R D3 R G3 R".split()
    )
]

# -------------------- Render per-instrument --------------------
def render_instrument(inst: Instrument, sr: int, token_dur: float) -> np.ndarray:
    parts = []
    for tok in inst.tokens:
        if tok == "R":
            parts.append(np.zeros(int(round(token_dur * sr)), dtype=np.float32))
        else:
            # octave shift: if token is note (e.g., E4) we add octshift to octave
            # handle numeric freq tokens too
            try:
                freq = note_to_freq(tok)
            except ValueError:
                # try appending octshift if token ends with digit(s)
                # fallback: treat as rest
                freq = 0.0
            # apply octave shift by doubling/halving
            if freq > 0.0 and inst.octshift != 0:
                freq *= (2.0 ** inst.octshift)
            s = synth_waveform(inst.wave, freq, token_dur, sr)
            # apply ADSR
            adsr = inst.adsr or {'attack':0.01,'decay':0.02,'sustain':0.9,'release':0.02}
            s = apply_adsr(s, sr, adsr['attack'], adsr['decay'], adsr['sustain'], adsr['release'])
            parts.append(s)
    if len(parts) == 0:
        return np.zeros(0, dtype=np.float32)
    inst_track = np.concatenate(parts).astype(np.float32)
    # apply gain
    gain_linear = 10.0 ** (inst.gain_db / 20.0)
    inst_track *= gain_linear
    # optional normalize (peak to -3dB)
    if inst.normalize and inst_track.size > 0:
        peak = np.max(np.abs(inst_track))
        if peak > 1e-9:
            target_peak = 10 ** (-3.0/20.0)  # -3 dB
            inst_track *= (target_peak/peak)
    return inst_track

# -------------------- Mixer --------------------
def mix_tracks(tracks: List[np.ndarray]) -> np.ndarray:
    # pad to same length
    maxlen = max((t.size for t in tracks), default=0)
    if maxlen == 0:
        return np.zeros(0, dtype=np.float32)
    mix = np.zeros(maxlen, dtype=np.float32)
    for t in tracks:
        if t.size == 0:
            continue
        mix[:t.size] += t
    # prevent clipping by scaling if needed
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix /= peak
    return mix

# -------------------- Main --------------------
def main():
    token_dur = DEFAULT_TOKEN_DUR
    print(f"SR={SR}, BPM={BPM}, token_dur={token_dur:.3f}s")
    rendered = []
    out_files = []
    os.makedirs("out", exist_ok=True)
    for inst in instruments:
        print(f"Rendering instrument: {inst.name} (wave={inst.wave}, gain_db={inst.gain_db}, octshift={inst.octshift})")
        track = render_instrument(inst, SR, token_dur)
        if track.size == 0:
            print(" - empty track")
            rendered.append(track)
            out_files.append(None)
            continue
        fname = os.path.join("out", f"{inst.name}.wav")
        write_wav_mono(fname, track, SR, BIT_DEPTH)
        print(f" - wrote {fname} (len={len(track)/SR:.2f}s)")
        rendered.append(track)
        out_files.append(fname)

    # mix
    print("Mixing...")
    mix = mix_tracks(rendered)
    # small final normalization to -0.5 dB
    peak = np.max(np.abs(mix)) if mix.size>0 else 0.0
    if peak > 1e-9:
        target = 10 ** (-0.5/20.0)
        mix *= (target / peak)
    final_path = os.path.join("out", FINAL_NAME)
    write_wav_mono(final_path, mix, SR, BIT_DEPTH)
    print(f"Final mix written: {final_path}")

    # list outputs
    print("Outputs:")
    for f in out_files:
        if f:
            print(" -", f)
    print(" -", final_path)

if __name__ == "__main__":
    main()
