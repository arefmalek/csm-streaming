from whisper import load_model
from pynvml import *
from enum import Enum
from dataclasses import dataclass
import wave

class ModelName(Enum):
    TINY = 0
    BASE = 1073741824
    SMALL = 2147483648
    MEDIUM = 5368709120
    LARGE = 10737418240

@dataclass
class DeviceData:
    index: int
    name: str
    type: str
    vram_free_bytes: int

# Create a new (.wav) audio file using two time stamps (in seconds, start & end)
def clipAudio(input_file, output_file, start, end):
    # Open the input .wav file
    with wave.open(input_file, 'rb') as wav_file:
        # Get the parameters of the input audio file
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        # Calculate the start and end frame positions based on the timestamps
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)
        # Clamp end_frame to last frame value in .wav
        end_frame = min(end_frame, num_frames)
        # Move the file pointer to the start_frame position
        wav_file.setpos(start_frame)
        # Read audio data from start_frame to end_frame
        audio_data = wav_file.readframes(end_frame - start_frame)

    # Write the clipped audio data to a new .wav file
    with wave.open(output_file, 'wb') as new_wav_file:
        new_wav_file.setnchannels(channels)
        new_wav_file.setsampwidth(sample_width)
        new_wav_file.setframerate(frame_rate)
        new_wav_file.writeframes(audio_data)

def get_best_device() -> DeviceData:
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    device = DeviceData(None, 'cpu', 'cpu', 0)
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        vram_free_bytes = nvmlDeviceGetMemoryInfo(handle).free
        if vram_free_bytes > device.vram_free_bytes:
            device = DeviceData(i, name, 'cuda', vram_free_bytes)
    nvmlShutdown()
    
    return device

def get_model_name(device: DeviceData) -> ModelName:
    # get device with most memory

    if device.vram_free_bytes >= ModelName.LARGE.value:
        return ModelName.LARGE
    elif device.vram_free_bytes >= ModelName.MEDIUM.value:
        return ModelName.MEDIUM
    elif device.vram_free_bytes >= ModelName.SMALL.value:
        return ModelName.SMALL
    elif device.vram_free_bytes >= ModelName.BASE.value:
        return ModelName.BASE
    else: 
        assert(ModelName.TINY.value <= device.vram_free_bytes < ModelName.BASE.value)
        return ModelName.TINY

def partition_file(input_file: str):
    device = get_best_device()
    model_name = get_model_name(device)
    model = load_model(model_name.name.lower(), device.type)
    
    result = model.transcribe(input_file)

    basename = input_file.split('.wav')[0]
    for i, segment in enumerate(result['segments']):
        clipAudio(input_file, f"{basename}_{i}.wav", segment['start'], segment['end'])
        print((segment['start'], segment['end']), segment['text'])

if __name__ == "__main__":
    partition_file("./audio_data/anna.wav")