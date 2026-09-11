"""
================================================================================
 Audio Course · Chapter 1 · 音频数据基础：波形 / 频谱 / 频谱图（学习笔记描述）
================================================================================
 一句话：搞懂"声音在计算机里长什么样"——时域波形、频域频谱、时频频谱图，是所有音频任务的地基。
 本章讲：
   ① 波形(waveform)：采样率 sampling_rate + 幅度值[-1,1]；用 librosa.load / waveshow 可视化。
   ② 频谱(spectrum)：对一小段做 DFT(np.fft.rfft) → 各频率的强度(dB)。
   ③ 频谱图(spectrogram)/梅尔谱：x=时间 y=频率，模型最常吃的输入表示。
 要点：音频要"看得见"才好调试(归一化/重采样/滤波是否生效、错样本长啥样)。
 说明：需 librosa/matplotlib；示例用内置 trumpet 音频，可本机跑出图。
================================================================================
"""

#pip install librosa
import librosa

#array该示例以音频时间序列（这里我们称之为 `x` ）和采样率（` r`）
# 的元组形式加载sampling_rate。让我们使用 librosa 的函数来查看此声音的波形waveshow()：
array,sampling_rate=librosa.load(librosa.ex("trumpet"))
import matplotlib.pyplot as plt
import librosa.display
plt.figure().set_figwidth(12)
librosa.display.waveshow(array,sr=sampling_rate)
# 此图以 y 轴表示信号幅度，x 轴表示时间。换句话说，
# 每个点对应于对该声音进行采样时所取的一个样本值。另请注意，librosa
# 已将音频以浮点值形式返回，并且幅度值确实在 [-1.0, 1.0] 范围内。
#
# 将音频可视化与聆听相结合，是理解所处理数据的有效工具。您可以观察信号的形状、
# 识别模式，并学会识别噪声或失真。如果您对数据进行预处理，例如归一化、重采样或滤波，
# 则可以直观地确认预处理步骤是否按预期执行。训练模型后，您还可以可视化出现错误的样本（例如在音频分类任务中）
# ，以便调试问题。

#频谱
#另一种可视化音频数据的方法是绘制音频信号的频谱图，也称为频域 表示。
# 频谱图是使用离散傅里叶变换 (DFT) 计算得出的。它描述了构成信号的各个频率及其强度。
import numpy as np
dft_input = array[: 4096 ] # 计算 DFT
window=np.hanning(len(dft_input))
windowed_input=dft_input*window
dft = np.fft.rfft(windowed_input) # 获取分贝幅度谱
amplitude = np. abs (dft)

amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max ) #获取频率区间
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft= len (dft_input))
plt.figure().set_figwidth( 12 )
plt.plot(frequency, amplitude_db)
plt.xlabel( "频率 (Hz)" )
plt.ylabel( "幅度 (dB)" )
plt.xscale( "log" )

#频谱图
import numpy as np
#x 轴表示时间，与波形可视化图相同，但 y 轴表示频率，单位为赫兹 (Hz)。
# 颜色强度表示每个时间点频率分量的振幅或功率，单位为分贝 (dB)。
D = librosa.stft(array)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar()

#梅尔光谱图
#n_mels表示要生成的梅尔频带数量。梅尔频带定义了一组频率范围，
# 这些范围使用一组滤波器将频谱划分为感知上有意义的成分，这些滤波器的形状和间距经过精心选择，
# 以模拟人耳对不同频率的响应方式。常用的值为n_mels40 或 80。fmax 表示我们关心的最高频率（单位为赫兹）
S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar()



#加载并浏览音频数据集
#pip install datasets[audio] soundfile librosa torchcodec
from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
print(minds)

example = minds[0]
print(example)
# 您可能会注意到音频栏目包含多个功能。以下是这些功能的具体内容：
# path：音频文件的路径（*.wav在本例中）。
# array解码后的音频数据，表示为一维 NumPy 数组。
# sampling_rate音频文件的采样率（本例中为 8,000 Hz）。
# 这intent_class是音频录制的分类类别。要将此数字转换为有意义的字符串，我们可以使用以下int2str()方法：

id2label = minds.features[ "intent_class" ].int2str
id2label(example[ "intent_class" ])

columns_to_remove = ["lang_id", "english_transcription"]
minds = minds.remove_columns(columns_to_remove)
minds

import gradio as gr


def generate_audio():
    example = minds.shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label(example["intent_class"])


with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)

import librosa
import matplotlib.pyplot as plt
import librosa.display

array = example["audio"]["array"]
sampling_rate = example["audio"]["sampling_rate"]

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate)


#音频数据集预处理

from datasets import Audio

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

#筛选数据集

MAX_DURATION_IN_SECONDS = 20.0

def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS

# 使用 librosa 从音频文件中获取示例的持续时间
new_column = [
    librosa.get_duration(y=x["array"], sr=x["sampling_rate"]) for x in minds["audio"]
]
minds = minds.add_column("duration", new_column)

# 使用 🤗 数据集的 `filter` 方法应用过滤函数
minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])

# 移除临时辅助列
minds = minds.remove_columns(["duration"])
minds
#音频数据预处理
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

def prepare_dataset(example):
    audio = example["audio"]

    if audio["sampling_rate"] != 16000:
        audio_array = librosa.resample(
            audio["array"], orig_sr=audio["sampling_rate"], target_sr=16000
        )
        audio = {"array": audio_array, "sampling_rate": 16000}

    features = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], padding=True
    )
    return features
minds = minds.map(prepare_dataset)
minds

import numpy as np

example = minds[0]
input_features = example["input_features"]

plt.figure().set_figwidth(12)
librosa.display.specshow(
    np.asarray(input_features[0]),
    x_axis="time",
    y_axis="mel",
    sr=feature_extractor.sampling_rate,
    hop_length=feature_extractor.hop_length,
)
plt.colorbar()

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-small")

#流媒体音频数据

gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", streaming=True)
next(iter(gigaspeech["train"]))

gigaspeech_head = gigaspeech["train"].take(2)
list(gigaspeech_head)