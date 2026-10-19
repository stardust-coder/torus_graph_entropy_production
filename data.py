from __future__ import annotations
import numpy as np
from human_eeg import *

def build_human_eeg_raw(
    patient_id: int = 2,
    state_name: str = "baseline",
    band: tuple[int, int] = (110, 140),
    relative_phase: bool = False
):

    def get_state_id(patient_id, state_name):
        state_names = HUMAN_EEG_RECORDINGS[patient_id].state_ids
        states = {"baseline":0, "mild": 1, "moderate": 2, "recovery": 3}
        return state_names[states[state_name]]

    human = load_human_eeg(
        patient_id=patient_id,
        state_id=get_state_id(patient_id, state_name),
        window=slice(0, 2000),
    )

    phase_spec = FeatureSpec(
        name="phase",
        feature="phase",
        band=band,
    )

    channel_indices = human_channel_indices(
        human,
        dim=19,
    )

    raw = extract_feature_matrix(
        human,
        phase_spec,
        trials=[0],
        channels=channel_indices,
    )

    if relative_phase:
        ### relative phase dynamicsを見る場合には以下を追加
        raw = raw[:, :-1] - raw[:, [-1]]
        raw = (raw + np.pi) % (2 * np.pi) - np.pi
    
    return raw

def get_phase_data(patient_id: int, band: str, state: str, value_type: str = 'absolute') -> np.ndarray:
    """Return phases in radians, with shape (N, d), N > lags.

    IMPORTANT: Implement the actual loading routine before running batch-ep.
    patient_id is an integer. value_type is 'absolute' or 'relative'.
    """
    bands = {"delta": (1,4), "theta":(4,8), "alpha": (8,15), "beta": (12,25), "gamma": (25,40), "hgamma": (40,110)}
    relative_phase = True if value_type == "relative" else False
    raw = build_human_eeg_raw(
        patient_id=patient_id, #これはstim_indexではなく、引数入力の際に中身はpatient_idとすることに注意.
        state_name=state,
        band=bands[band],
        relative_phase=relative_phase    ### relative phase dynamicsを見る場合にはTrue. 最後の1次元を残りの次元から差っ引いた上でdropします。
    )
    return raw