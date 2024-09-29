import os
import torch
import librosa
import look2hear.models
import soundfile as sf
from tqdm.auto import tqdm
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def load_audio(file_path):
    audio, samplerate = librosa.load(file_path, mono=False, sr=44100)
    print(f'INPUT audio.shape = {audio.shape} | samplerate = {samplerate}')
    return torch.from_numpy(audio), samplerate

def save_audio(file_path, audio, samplerate=44100):
    sf.write(file_path, audio.T, samplerate, subtype="PCM_16")

def process_chunk(chunk):
    chunk = chunk.unsqueeze(0).cuda()
    with torch.no_grad():
        return model(chunk).squeeze(0).squeeze(0).cpu()

def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(1, 1, fade_size)
    fadeout = torch.linspace(0, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def dBgain(audio, volume_gain_dB):
    gain = 10 ** (volume_gain_dB / 20)
    gained_audio = audio * gain
    return gained_audio

def process_audio_file(input_wav, output_wav, ckpt_path):
    global model
    model = look2hear.models.BaseModel.from_pretrain(ckpt_path, sr=44100, win=20, feature_dim=256, layer=6).cuda()

    test_data, samplerate = load_audio(input_wav)

    C = chunk_size * samplerate
    N = overlap
    step = C // N
    fade_size = 3 * 44100
    print(f"N = {N} | C = {C} | step = {step} | fade_size = {fade_size}")

    border = C - step

    if len(test_data.shape) == 1:
        test_data = test_data.unsqueeze(0)

    if test_data.shape[1] > 2 * border and (border > 0):
        test_data = torch.nn.functional.pad(test_data, (border, border), mode='reflect')

    windowingArray = _getWindowingArray(C, fade_size)

    result = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)
    counter = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)

    i = 0
    progress_bar = tqdm(total=test_data.shape[1], desc="Processing audio chunks", leave=False)

    while i < test_data.shape[1]:
        part = test_data[:, i:i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
            else:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

        out = process_chunk(part)

        window = windowingArray
        if i == 0:
            window[:fade_size] = 1
        elif i + C >= test_data.shape[1]:
            window[-fade_size:] = 1

        result[..., i:i+length] += out[..., :length] * window[..., :length]
        counter[..., i:i+length] += window[..., :length]

        i += step
        progress_bar.update(step)

    progress_bar.close()

    final_output = result / counter
    final_output = final_output.squeeze(0).numpy()
    np.nan_to_num(final_output, copy=False, nan=0.0)

    if test_data.shape[1] > 2 * border and (border > 0):
        final_output = final_output[..., border:-border]

    save_audio(output_wav, final_output, samplerate)
    print(f'Success! Output file saved as {output_wav}')

    model.cpu()
    del model
    torch.cuda.empty_cache()

def main(input_dir, output_dir, ckpt_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            input_wav = os.path.join(input_dir, file_name)
            output_wav = os.path.join(output_dir, file_name)
            process_audio_file(input_wav, output_wav, ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument("--in_dir", type=str, required=True, help="Path to input directory containing wav files")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to output directory for processed wav files")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint file", default="model/pytorch_model.bin")
    parser.add_argument("--chunk_size", type=int, help="chunk size value in seconds", default=10)
    parser.add_argument("--overlap", type=int, help="Overlap", default=2)
    args = parser.parse_args()

    ckpt_path = args.ckpt
    chunk_size = args.chunk_size
    overlap = args.overlap
    print(f'ckpt_path = {ckpt_path}')
    print(f'chunk_size = {chunk_size}, overlap = {overlap}')

    main(args.in_dir, args.out_dir, ckpt_path)
