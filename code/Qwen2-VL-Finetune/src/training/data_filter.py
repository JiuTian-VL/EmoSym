import sys
import os
import shutil
import json
import requests
import numpy as np
import torch
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdb
sys.path.append("/data7/Users/zyj/TAPMI/EmoGen")
from model import *
import argparse
import random
def parse_args():
    parser = argparse.ArgumentParser(description="Run emotion-based processing")
    parser.add_argument('--output_base', type=str, required=True, 
                        help="The base directory for output files (image, annotation, json).")
    return parser.parse_args()
def get_sample(scores:list) -> int:
    """
    Args:
        scores: [score_1, score_2, ... , score_n]
    Return:
        The index of the chosen sample.
    """
    total_score = sum(scores)
    r = random.uniform(0, total_score)
    now_score = 0
    for i, score in enumerate(scores):
        now_score += score
        if now_score >= r:
            return i

def get_samples(all_samples:list, n_sample:int) -> list:
    """
    Args:
        all_sample: [(score_1, 'str_1'), (score_2, 'str_2'), ... , (score_n, 'str_n')]
        n_sample: The number of samples to be extracted.
    Return:
        A list of n_sample extracted samples, ['str_i', ... , 'str_j'] 
    """
    all_samples = [(score*1000, Str) for score, Str in all_samples]
    all_samples = sorted(all_samples, key= lambda x: x[0], reverse=True)
    all_score = [score for score, _ in all_samples]
    all_str = [Str for _, Str in all_samples]
    print(all_str[:10])
    samples = []
    for _ in range(n_sample):
        index = get_sample(all_score)
        samples.append(all_str[index])
        del all_score[index]
        del all_str[index]

    return samples

# test = [(4.158339672556815e-05, 'amusement_13004.jpg'), (3.0312633888913338e-05, 'amusement_04315.jpg'), (3.949716899261248e-05, 'amusement_09033.jpg'), (5.7840950763143706e-05, 'amusement_02279.jpg'), (5.2815080754945376e-05, 'amusement_4571_strawberry_21.34_v2.jpg'), (5.3832234332678366e-05, 'amusement_1932_amusement_park_22.22_v2.jpg'), (5.129407781123237e-05, 'amusement_5337_amusement_park_24.16_v2.jpg'), (2.76442785999034e-05, 'amusement_16909.jpg'), (5.541870877543997e-05, 'amusement_1579_picnic_area_20.92_v2.jpg'), (5.769661328517737e-05, 'amusement_02280.jpg'), (4.603231815143908e-05, 'amusement_02834.jpg'), (3.504269472999053e-05, 'amusement_12837.jpg'), (3.6659239714318466e-05, 'amusement_05545.jpg'), (3.799311877474156e-05, 'amusement_4292_bird_23.78_v2.jpg'), (5.6138122573892114e-05, 'amusement_16173.jpg'), (2.9423106971880013e-05, 'amusement_09528.jpg'), (4.259793109511618e-05, 'amusement_394_playground_19.84_v2.jpg'), (4.329850248811961e-05, 'amusement_2093_nursing_home_20.21_v2.jpg'), (2.250352297372499e-05, 'amusement_19154.jpg'), (3.157724930589711e-05, 'amusement_07502.jpg'), (4.3037709390400766e-05, 'amusement_1840_amusement_park_23.04_v2.jpg'), (2.549307393754352e-05, 'amusement_18720.jpg'), (3.6687592755635526e-05, 'amusement_11877.jpg'), (3.127575537492006e-05, 'amusement_2788_sports_uniform_23.76_v2.jpg'), (2.9342979717360707e-05, 'amusement_11279.jpg'), (3.785178748248793e-05, 'amusement_08541.jpg'), (4.999178533834442e-05, 'amusement_09993.jpg'), (5.032985123984145e-05, 'amusement_953_amusement_park_24.42_v2.jpg'), (4.026982525625061e-05, 'amusement_13379.jpg'), (4.450609514430113e-05, 'amusement_04807.jpg'), (3.138208682312723e-05, 'amusement_04773.jpg'), (3.865077685566182e-05, 'amusement_07576.jpg'), (5.5065004609478864e-05, 'amusement_02116.jpg'), (4.2589530298100694e-05, 'amusement_344_christmas_tree_22.28_v2.jpg'), (5.1489740557872154e-05, 'amusement_13731.jpg'), (2.7649794973956875e-05, 'amusement_16049.jpg'), (3.0853086993530913e-05, 'amusement_07172.jpg')]
# print(get_samples(test, 5))
# Define emotion list and transformations
emotion_list_8 = {
    "amusement": 0,
    "awe": 1,
    "contentment": 2,
    "excitement": 3,
    "anger": 4,
    "disgust": 5,
    "fear": 6,
    "sadness": 7
}

tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


emotion_before_num = {
      "amusement": 19445,
      "awe": 15037,
      "contentment": 16337,
      "excitement": 19828,
      "anger": 10660,
      "disgust": 10660,
      "fear": 13453,
      "sadness": 12676
    }
def generate_train_json(image_folder, annotation_folder, output_json):
    data_list = []

    # Traverse the image folder
    for root, _, files in os.walk(image_folder):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract emotion from the path
                relative_image_path = os.path.relpath(os.path.join(root, file_name), image_folder)
                emotion = relative_image_path.split('/')[0]

                # Construct the corresponding annotation file path
                base_name = os.path.splitext(file_name)[0]
                annotation_file_name = base_name + '.json'
                relative_annotation_path = os.path.join(annotation_folder, emotion, annotation_file_name)
                # pdb.set_trace()
                # Check if the corresponding annotation file exists
                if os.path.exists(relative_annotation_path):
                    data_list.append([
                        emotion,
                        os.path.join('image', relative_image_path),
                        os.path.relpath(relative_annotation_path, os.path.dirname(output_json))
                    ])
                else:
                    print(f"Warning: Annotation file not found for image {relative_image_path}")

    # Write to the output JSON file
    with open(output_json, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)
    print(len(data_list))
    print(f"Data successfully written to {output_json}")
def create_annotations(image_dir, annotation_dir):
    # Iterate over each emotion category in the image directory
    for emotion in os.listdir(image_dir):
        emotion_path = os.path.join(image_dir, emotion)
        if os.path.isdir(emotion_path):
            # Create an annotation subdirectory for this emotion
            annotation_subdir = os.path.join(annotation_dir, emotion)
            if not os.path.exists(annotation_subdir):
                os.makedirs(annotation_subdir)
            
            # Iterate over each image in this emotion subfolder
            for img_file in os.listdir(emotion_path):
                if img_file.endswith('.jpg'):  # Assuming the images are jpg
                    img_id = img_file.replace('.jpg', '')
                    annotation_data = {
                        "image_id": img_id,
                        "emotion": emotion
                    }
                    # Define the path for the annotation JSON file
                    annotation_path = os.path.join(annotation_subdir, f"{img_id}.json")
                    
                    # Write the annotation to a JSON file
                    with open(annotation_path, 'w') as json_file:
                        json.dump(annotation_data, json_file, indent=4)
if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    # 
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    folders = [
        '/zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/img/amusement',
        '/zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/img/anger',
        '/zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/img/awe',
        '/zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/img/contentment',
        '/zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/img/disgust',
        '/zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/img/excitement',
        '/zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/img/fear',
        '/zhu_yi_jie/Zyj_MM/joint_train/context_RL_12/1/img/sadness'
    ]
 
    # pdb.set_trace()
    output_base = args.output_base
    image_folder = os.path.join(output_base, 'image')
    annotation_folder = os.path.join(output_base, 'annotation')

    create_annotations(image_folder, annotation_folder)
    output_json = os.path.join(output_base, 'train.json')

    generate_train_json(image_folder, annotation_folder, output_json)
    info_json_target = os.path.join(output_base, 'info.json')
    shutil.copy('/zhu_yi_jie/Zyj_MM/EmoSet_aug_context/info.json', info_json_target)
   



