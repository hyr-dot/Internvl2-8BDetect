import numpy as np
import torch
import torchvision.transforms as T
import os
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import warnings
import time
import json
import ipdb
import random
import albumentations
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用GPU 0
device = torch.device('cuda:0')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

path = '/home/ybw/hyr_temp/AIdetc/analyzers/internvl'
config = InternVLChatConfig.from_pretrained(path)
model = InternVLChatModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    config = config,
).to(device)
print("Model loaded on:", next(model.parameters()).device)

# model = model.to(torch.device('cuda:1'))
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# set the max number of tiles in `max_num`
# pixel_values = load_image('/home/ybw/hyr_temp/InternVL2-8B/temp1/0/ad0611c95a23c2738c1fd7ab6fd2a1e9.jpg', max_num=6).to(torch.bfloat16).cuda()

generation_config = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
    
)

device_map = {
    'vision_model': 1,
    'mlp1': 1,
    'language_model.model.tok_embeddings': 1,  # near the first layer of LLM
    'language_model.model.norm': 1,  # near the last layer of LLM
    'language_model.output.weight': 1  # near the last layer of LLM
}
for i in range(16):
    device_map[f'language_model.model.layers.{i}'] = 1
for i in range(16, 48):
    device_map[f'language_model.model.layers.{i}'] = 1

def video_test(filepath):
    start_time = time.time()
    if filepath.endswith('.mp4'):
        pixel_values, num_patches_list = load_video(filepath, num_segments=8, max_num=1)    
        pixel_values = pixel_values.to(torch.bfloat16).to(device)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        # ipdb.set_trace()
        question = video_prefix + 'Tell me if there are synthesis artifacts in the video or not. Must return with: 1)yes or no only. 2)If yes, explain where the artifacts is with one sentence.'
        # Frame1: <image>\nFrame2: <image>\n...\nFrame31: <image>\n{question}
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
           num_patches_list=num_patches_list,
           history=None, return_history=True)
    elif filepath.endswith('.png') or filepath.endswith('.jpg'):
        pixel_value = load_image(filepath, max_num=6).to(torch.bfloat16).to(device)
        question = '<image>\nTell me if there are synthesis artifacts in the image or not. Must return with: 1)yes or no only. 2)If yes, explain where the artifacts is with one sentence.'
        response, _ = model.chat(tokenizer, pixel_value, question, generation_config, history=None, return_history=True)
    print(f'target:{filepath}')
    print(f'Assistant: {response}')
    finish_time = time.time()
    print('time cost is:', finish_time - start_time, 's')
if __name__ == '__main__':
    # video_test("/mnt/data/AIGC/test/fake/gen2_14.mp4")
    dir = "/mnt/data/hyr_files/xian_test/test_fei"
    for image in os.listdir(dir):
        img_path = os.path.join(dir, image)
        video_test(img_path)
    
