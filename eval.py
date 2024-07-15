import os
import cv2
import re
import time
import psutil
import pynvml
import tracemalloc
from paddleocr import PaddleOCR


def wer(word1, word2):  # 计算wer
    # 先计算编辑距离然后除以word1长度得到wer
    m, n = len(word1), len(word2)

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        current = [i] * (n + 1)
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                current[j] = prev[j - 1]
            else:
                current[j] = min(prev[j - 1], current[j - 1], prev[j]) + 1
        prev = current

    return prev[-1] / m


def txt_to_dict(file_path):  # 将label的txt文件转换成字典提高查询速度
    with open(file_path, 'r', encoding='utf-8') as file:
        return {line.split('\t')[0]: line.split('\t')[1].strip() for line in file}


def get_gpu_usage():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / (1024 ** 2)  # 以MB为单位
    except:
        return 0  # 未使用GPU时返回0


images_path = "data/ai_img"  # 图片文件夹路径（替换为自己的路径）
labels_path = "data/ai_img/Label.txt"  # 标签文件路径（替换为自己的路径）
label_dict = txt_to_dict(labels_path)

ocr = PaddleOCR(cpu_threads=20, max_batch_size=20, rec_batch_num=20, precision='bf16', rec_image_inverse=False, use_space_char=False)  # 加载ocr模型
all_wer = []  # 各图片的wer
num_image = 0  # 图片数量

start_time = time.time()
tracemalloc.start()

with open(labels_path, 'r', encoding='utf-8') as file:
    all_files = os.listdir(images_path)
    for filename in [filename for filename in all_files if filename.endswith(('.jpeg', '.jpg', '.png'))]:
        try:
            image_path = os.path.join(images_path, filename)  # 获取图片路径
            image = cv2.imread(image_path)  # 读取图片
            if image.shape[0] > 640:
                image = cv2.resize(image, (int(image.shape[1] * 640 / image.shape[0]), 640), interpolation=cv2.INTER_AREA)

            # 获取ocr结果
            ocr_results = ocr.ocr(image)
            ocr_result = ''.join([result[1][0] for result in ocr_results[0]])

            # 获取label
            image_labels = label_dict.get(filename, "")
            transcriptions = re.findall(r'"transcription": "(.*?)"', image_labels)
            image_label = ''.join(transcriptions)

            num_image += 1
            all_wer.append(wer(image_label, ocr_result))
        except:
            continue

end_time = time.time()
cpu_usage = psutil.cpu_percent()
memory_info = psutil.virtual_memory()
gpu_usage = get_gpu_usage()
_, peak_memory = tracemalloc.get_traced_memory()
tracemalloc.stop()

average_wer = sum(all_wer) / num_image  # 平均wer
print("average_wer:", average_wer)
print("accuracy:", (1 - average_wer) * 100, "%")  # 平均正确率
print("CPU Usage:", cpu_usage, "%")
print("Memory Usage:", memory_info.used / (1024 ** 2), "MB")
print("GPU Usage:", gpu_usage, "MB")
print("Peak Memory Usage:", peak_memory / (1024 ** 2), "MB")
print("Time:", end_time - start_time, "s")
print("PPS:", num_image / (end_time - start_time))
