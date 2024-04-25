from flask import Flask, request, jsonify
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
import os
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
import traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from skimage.transform import resize


app = Flask(__name__)

def load_tflite_model(tflite_path):
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


interpreter = load_tflite_model('./model.tflite')

category_features = {}
categories = ['Rooster', 'Frog', 'Dog', 'Cow', 'Pig']  # 示例类别

def load_features_numpy(file_path):
    # 加载 .npy 文件，allow_pickle=True 允许加载对象数组
    data = np.load(file_path, allow_pickle=True)
    
    # 如果数据是被保存为数组而非字典，这一步会将其转换回字典
    return data.item() if isinstance(data, np.ndarray) else data

category_features = load_features_numpy('features.npy')  # 加载特征的函数

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.wav'):
        # 保存到临时目录
        filename = os.path.join('/tmp', file.filename)
        file.save(filename)
        
        # 将文件名（不含扩展名）作为指定的动物类别
        specified_category = os.path.splitext(file.filename)[0]
        
        # 使用模型处理音频并计算相似度
        try:
            similarity_percent = similarity_to_specified_animal(interpreter, filename, specified_category, category_features)
            score = getScore(similarity_percent)
            return jsonify({'Your score': score})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

def process_and_identify(audio_path):
    # 这里应当调用一个函数来处理音频文件并使用模型进行识别
    # 返回识别结果
    return {"animal": "detected_animal", "similarity": "95%"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

def getScore(percent):
    if percent < 1:
        return 1
    elif percent < 5:
        return 1
    elif percent < 10:
        return 2
    elif percent < 15:
        return 3
    elif percent < 25:
        return 4
    elif percent < 40:
        return 5
    elif percent < 50:
        return 6
    elif percent < 80:
        return 7
    elif percent < 90:
        return 8
    elif percent < 99.9:
        return 9
    else:
        return 10

def load_and_segment_audio(audio_path, target_length=1.5):
    y, sr = librosa.load(audio_path)
    buffer_length = int(sr * target_length)
    segments = [y[i:i + buffer_length] for i in range(0, len(y), buffer_length) if i + buffer_length <= len(y)]
    return segments, sr



def add__noise(data_segment, noise_level=0.005):
    # Ensure the noise is generated with the same shape as the data segment
    noise = np.random.randn(*data_segment.shape)
    augmented_data_segment = data_segment + noise_level * noise
    return augmented_data_segment


def resize_melspectrogram(mels, target_shape=(128, 128)):
    # 使用 skimage 的 resize 函数调整 mels 的大小
    return resize(mels, target_shape, mode='constant', anti_aliasing=True)

def extract_melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_resized = resize_melspectrogram(S_DB, target_shape=(128, 128))
                # 确保增加一个维度来表示单通道
    S_resized = S_resized[..., np.newaxis]
    return S_resized


def process_and_visualize(audio_path, target_length=1.5, noise_level=0.005):
    # Load and segment audio
    segments, sr = load_and_segment_audio(audio_path, target_length=target_length)
    
    processed_segments = []
    for segment in segments:
        # Add noise to the individual segment
        noisy_segment = add__noise(segment, noise_level=noise_level)
        
        # Extract mel spectrogram
        melspectrogram = extract_melspectrogram(noisy_segment, sr)
        
        processed_segments.append(melspectrogram)
    
    # If needed, visualize or further process the segments
    return processed_segments


# 示例：处理并可视化音频文件
#melspectrogram = process_and_visualize(str(audio_file))

def extract_features(interpreter, audio_path):
    # 获取输入和输出详细信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 处理并可视化音频以获得梅尔频谱图
    spectrograms = process_and_visualize(audio_path)
    if spectrograms:
        # 使用第一个频谱图，或者可以选择其他方法如平均频谱图
        melspectrogram = spectrograms[0]

        # 确保输入数据符合输入张量的期望形状
        input_shape = input_details[0]['shape']
        # 重要的类型转换步骤
        melspectrogram = np.resize(melspectrogram, input_shape).astype(np.float32)

        # 设置输入张量
        interpreter.set_tensor(input_details[0]['index'], melspectrogram)

        # 运行模型
        interpreter.invoke()

        # 从输出张量中获取特征
        features = interpreter.get_tensor(output_details[0]['index'])

        return features
    else:
        raise ValueError("No spectrograms generated from the audio processing.")


# 相似度计算函数
def calculate_similarity(feature1, feature2):
    return cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))[0][0]


def similarity_to_specified_animal(model, audio_path, specified_category, category_features):
    # 提取音频的特征
    new_audio_features = extract_features(model, audio_path)

    # 检查指定的类别是否存在于特征字典中
    if specified_category not in category_features:
        raise ValueError(f"Specified category '{specified_category}' not found in the provided categories.")

    # 计算与指定类别的特征的相似度
    cat_feat = category_features[specified_category]
    similarity = calculate_similarity(new_audio_features, cat_feat)

    # 转换为百分比形式
    similarity_percent = similarity * 100

    print(f"Similarity to {specified_category}: {similarity_percent:.2f}%")
    return similarity_percent
