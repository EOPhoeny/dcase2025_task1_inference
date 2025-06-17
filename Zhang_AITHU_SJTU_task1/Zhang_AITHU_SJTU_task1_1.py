"""
DCASE 2025 full fine-tuned baseline model — Modular API for ASC inference.
"""

from typing import Optional, List
import torch
import torchaudio
from torch import Tensor
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import importlib.resources as pkg_resources
from nni.compression.speedup import ModelSpeedup

# Model and resource imports
from Zhang_AITHU_SJTU_task1.models.baseline import get_model
from Zhang_AITHU_SJTU_task1 import ckpts


class Config:
    """Configuration for audio preprocessing and model structure."""

    # Audio parameters
    orig_sample_rate = 44100
    sample_rate = 32000

    # Spectrogram parameters
    window_length = 3072
    hop_length = 500
    n_fft = 4096
    n_mels = 256
    f_min = 0
    f_max = None

    # Model architecture
    n_classes = 10
    in_channels = 1
    base_channels = 64
    n_blocks = (3, 2, 1)
    channels_multiplier = 1.8
    expansion_rate = 2.2
    model_arch = 'baseline2'
    strides = 0

    batch_size = 256  # Batch size for inference


class Baseline2(torch.nn.Module):
    def __init__(self, config=Config, pruned_model=None):
        super().__init__()
        self.config = config

        self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.window_length,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max
        )

        if config.strides == 0:
            strides = None
        elif config.strides == 1:
            strides = dict(
                b2=(1, 1),
                b3=(1, 2),
                b4=(2, 1)
            )
        else:
            strides = None

        # the baseline model
        if pruned_model is None:
            self.model = get_model(n_classes=config.n_classes,
                                   in_channels=config.in_channels,
                                   base_channels=config.base_channels,
                                   channels_multiplier=config.channels_multiplier,
                                   expansion_rate=config.expansion_rate,
                                   n_blocks=config.n_blocks,
                                   model_arch=config.model_arch,
                                   n_mels=config.n_mels,
                                   strides=strides,
                                   )
        else:
            self.model = pruned_model

        self.model.eval()
        self.model.half()

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        self.class_order = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        x = (x + 1e-5).log()
        x = x.half()
        return x

    def forward(self, x) -> Tensor:
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        with torch.no_grad():
            x = self.mel_forward(x)
            x = self.model(x)
        return x


def load_backbone_model(config=Config):

    backbone_model = get_model(n_classes=config.n_classes,
                               in_channels=config.in_channels,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier,
                               expansion_rate=config.expansion_rate,
                               n_blocks=config.n_blocks,
                               model_arch=config.model_arch,
                               n_mels=config.n_mels,
                               )

    with pkg_resources.path(ckpts, "c64_unpruned_backbone.ckpt") as ckpt_path:
        model_path = str(ckpt_path)

    ckpt = torch.load(model_path, map_location="cpu")['state_dict']
    new_dict = {}
    for key in ckpt.keys():
        if 'mel.' in key or 'teacher' in key:
            continue
        elif 'model.' in key:
            new_key = key.replace("model.", "")
            new_dict[new_key] = ckpt[key]
        else:
            new_dict[key] = ckpt[key]
    backbone_model.load_state_dict(new_dict)
    return backbone_model


def load_model(model_file_path: Optional[str] = None,
               mask_file_path: Optional[str] = None) -> Baseline2:
    """
    Load the full fine-tuned baseline model from a checkpoint.

    Args:
        model_file_path: Optional path to a .ckpt file. If None, uses the default packaged checkpoint.

    Returns:
        A Baseline model instance with loaded weights.
    """
    config = Config()

    if model_file_path is None:
        # Use default checkpoint from package resources
        with pkg_resources.path(ckpts, "c64_unpruned_distilled_1.ckpt") as ckpt_path:
            model_file_path = str(ckpt_path)

    if mask_file_path is None:
        # Use default checkpoint from package resources
        with pkg_resources.path(ckpts, "c64_mask_1.pth") as mask_path:
            mask_file_path = str(mask_path)

    backbone_model = load_backbone_model(config)
    masks = torch.load(mask_file_path, map_location="cpu")
    dummy_input = torch.rand(config.batch_size, 1, 256, 65)  # [B, 1, n_mels, T]
    pruned_model = ModelSpeedup(backbone_model,
                                dummy_input, masks).speedup_model()

    # Load checkpoint
    ckpt = torch.load(model_file_path, map_location="cpu")['state_dict']

    new_dict = {}
    for key in ckpt.keys():
        if 'mel.' in key or 'teacher' in key:
            continue
        elif 'model.' in key:
            new_key = key.replace("model.", "")
            new_dict[new_key] = ckpt[key]
        else:
            new_dict[key] = ckpt[key]
    pruned_model.load_state_dict(new_dict)
    pruned_model.half()

    model = Baseline2(config, pruned_model)

    return model


def load_inputs(
    file_paths: List[str],
    device_ids: List[str],
    model: Baseline2,
    num_workers: int = 16,
    batch_size: int = 256
) -> List[Tensor]:
    """
    Load and preprocess audio files in parallel with batch-wise resampling and STFT.

    Args:
        file_paths: List of .wav file paths.
        device_ids: List of corresponding device IDs (same length as file_paths).
        model: Baseline model (used for preprocessing).
        num_workers: Number of threads used for parallel loading.
        batch_size: Number of waveforms per batch for STFT/mel processing.

    Returns:
        List of mel spectrogram tensors [1, 1, n_mels, T], in same order as file_paths.
    """
    assert len(file_paths) == len(device_ids)

    device = next(model.parameters()).device
    target_sr = model.config.sample_rate

    def _load(indexed_path):
        path, idx = indexed_path
        waveform, sr = torchaudio.load(path)              # [channels, samples]
        waveform = waveform.mean(dim=0)                   # mono: [samples]
        return idx, waveform, sr

    # Step 1: Load & mono-mix in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        loaded = list(tqdm(
            exe.map(_load, zip(file_paths, range(len(file_paths)))),
            total=len(file_paths),
            desc="Loading files"
        ))
    # loaded = list of (original_idx, waveform, sample_rate)

    # Step 2: Group by original sample rate to batch resampling
    sr_groups = defaultdict(list)
    for idx, waveform, sr in loaded:
        sr_groups[sr].append((idx, waveform))

    inputs: List[Tensor] = [None] * len(file_paths)  # final output buffer

    # Step 3: Batch resample and preprocess each group
    print("Batched Resampling ...")
    for sr, items in sr_groups.items():
        print(f"Processing SR={sr} with {len(items)} files")
        for i in tqdm(range(0, len(items), batch_size), desc=f"SR={sr}", leave=False):
            chunk = items[i:i + batch_size]
            indices, waveforms = zip(*chunk)

            # Pad waveforms to the same length → [max_len, B]
            padded = pad_sequence(waveforms, batch_first=False)

            # Reshape → [B, 1, samples]
            batch_wave = padded.transpose(0, 1).unsqueeze(1).to(device)

            # Resample if needed
            if sr != target_sr:
                batch_wave = torchaudio.functional.resample(
                    batch_wave, orig_freq=sr, new_freq=target_sr
                )

            # STFT + mel computation for batch
            with torch.no_grad():
                mel_batch = model.mel_forward(batch_wave)  # [B, 1, n_mels, T]

            # Store each preprocessed mel in original order
            for mel, idx in zip(mel_batch, indices):
                inputs[idx] = mel.unsqueeze(0).cpu()  # [1, 1, n_mels, T]

    return inputs


def get_model_for_device(
    model: Baseline2,
    device_id: str
) -> torch.nn.Module:
    """
    Extract the device model corresponding to a specific device ID.

    Args:
        model: Baseline model instance.
        device_id: Device identifier string (e.g., 's1').

    Returns:
        The device model (nn.Module) associated with the given device.
    """
    device_model = model.model  # No device-specific models
    device_model.half()
    return device_model


def predict(
    file_paths: List[str],
    device_ids: List[str],
    model_file_path: Optional[str] = None,
    use_cuda: bool = True,
    batch_size: int = 64
) -> List[torch.Tensor]:
    """
    Run inference on a list of audio files using device-specific models.

    Files are grouped by device ID and processed in batches.

    Args:
        file_paths: List of audio file paths.
        device_ids: List of device IDs corresponding to each file.
        model_file_path: Optional path to a model checkpoint (.ckpt).
        use_cuda: Whether to use GPU (if available).
        batch_size: Number of examples per inference batch.

Returns:
    A tuple (logits, class_order)
    - logits: List of tensors, one per file, each of shape [n_classes]
    - class_order: List of class names corresponding to the output logits
    """
    assert len(file_paths) == len(device_ids), "Number of files and device IDs must match."

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = load_model(model_file_path)

    # Step 1: Preprocess inputs → list of [1, 1, n_mels, T]
    inputs = load_inputs(file_paths, device_ids, model)

    # Step 2: Group by device ID, squeeze each mel to [n_mels, T]
    groups = defaultdict(list)
    for idx, (mel, dev) in enumerate(zip(inputs, device_ids)):
        mel_squeezed = mel.squeeze(0).squeeze(0)  # → [n_mels, T]
        groups[dev].append((mel_squeezed, idx))

    outputs = [None] * len(inputs)  # Placeholder for final predictions

    # Step 3: For each device, batch and infer
    for dev, items in tqdm(groups.items(), desc="Batched inference"):
        submodel = get_model_for_device(model, dev)

        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            mels, indices = zip(*chunk)  # List of [n_mels, T_i]

            # Pad to same length → [max_T, batch, n_mels]
            # This is to potentially also support files of varying length. However,
            # all files in the TAU dataset are exactly one second in length.
            padded = pad_sequence([m.T for m in mels], batch_first=False)

            # Reshape → [batch, 1, n_mels, max_T]
            batch = padded.permute(1, 2, 0).unsqueeze(1).to(device)
            with torch.no_grad():
                submodel = submodel.to(device)
                logits = submodel(batch)  # [B, n_classes]

            # Scatter outputs back in original file order
            for logit, idx in zip(logits, indices):
                outputs[idx] = logit

    return outputs, model.class_order
