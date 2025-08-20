import os, sys

import argparse
import urllib.request

import math
import cameralib
import numpy as np
import posepile.joint_info
import poseviz
import simplepyutils as spu
import torch
import torchvision.io
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.metrabs_models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config

from tqdm import tqdm

model_dir = ''

# TODO: Add Dataloader
def extract_poses(video_filepath, model, skeleton):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read video frames using torchvision
    MAX_FRAMES = 300
    frames, _, _ = torchvision.io.read_video(video_filepath, pts_unit="sec")
    current_frames = frames.shape[0]
    if current_frames > MAX_FRAMES:
        frames = frames[:MAX_FRAMES]  # Keep only the first MAX_FRAMES frames
    frames = frames.to(device)
    frames = frames.permute(0,3,1,2)

    CHANNELS = frames.shape[1]
    HEIGHT = frames.shape[2]
    WIDTH = frames.shape[3]

    MAX_DETECTIONS = 2

    batch_size = 256 # Only in the case that the video is too long, otherwise video is processed at once
    current_frames = frames.shape[0]

    with torch.inference_mode(), torch.device('cuda'):
        padded_poses = []
        if current_frames > batch_size:
            batches = math.ceil(current_frames/batch_size)
            for i in range (0, batches):
                # print("Frames:",len(frames[i*batch_size:(i+1)*batch_size]))
                batch_preds = model.detect_poses_batched(
                    frames[i*batch_size:(i+1)*batch_size], detector_threshold=0.01, suppress_implausible_poses=False,
                    max_detections=MAX_DETECTIONS, skeleton=skeleton)

                for pose in batch_preds['poses3d']:
                    current_size = len(pose)
                    padding_size = MAX_DETECTIONS - current_size
                    
                    # TODO: Make padding dependent on skeleton convention
                    # Create a padding tensor of shape [padding_size, 25, 3] with zeros
                    if padding_size > 0:
                        padding = torch.zeros((padding_size, 42, 3))
                        pose_tensor = torch.cat([torch.tensor(pose), padding], dim=0)
                    else:
                        pose_tensor = torch.tensor(pose)
                        
                    padded_poses.append(pose_tensor)

        else: 
            batch_preds = model.detect_poses_batched(
                    frames, detector_threshold=0.01, suppress_implausible_poses=False,
                    max_detections=MAX_DETECTIONS, skeleton=skeleton)

            for pose in batch_preds['poses3d']:
                current_size = len(pose)
                padding_size = MAX_DETECTIONS - current_size
                
                # Create a padding tensor of shape [padding_size, 25, 3] with zeros
                if padding_size > 0:
                    padding = torch.zeros((padding_size, 42, 3)) # TODO: Make dynamic 
                    pose_tensor = torch.cat([torch.tensor(pose), padding], dim=0)
                else:
                    pose_tensor = torch.tensor(pose)
                    
                padded_poses.append(pose_tensor)

    # Stack the padded poses into a single tensor
    preds = torch.stack(padded_poses)

    if frames.shape[0] != preds.shape[0]:
        raise RuntimeError("Frames and poses do not have the same first dimension.")
    #preds = torch.stack(batch_preds['poses3d'])
    del frames
    del padded_poses

    # Not returning the padded tensor actually --> Still need to pad them all to the same size before passing to ST-GCN
    return preds 

def load_multiperson_model():
    model_pytorch = load_crop_model()
    skeleton_infos = spu.load_pickle(f'{model_dir}/skeleton_infos.pkl')
    joint_transform_matrix = np.load(f'{model_dir}/joint_transform_matrix.npy')

    with torch.device('cuda'):
        return multiperson_model.Pose3dEstimator(
            model_pytorch.cuda(), skeleton_infos, joint_transform_matrix)


def load_crop_model():
    cfg = get_config()
    # cfg = cfg['metrabs_eff2l_384px_800k_28ds_pytorch']
    ji_np = np.load(f'{model_dir}/joint_info.npz')
    ji = posepile.joint_info.JointInfo(ji_np['joint_names'], ji_np['joint_edges'])
    backbone_raw = getattr(effnet_pt, f'efficientnet_v2_{cfg.efficientnet_size}')()
    # backbone_raw = getattr(effnet_pt, f'efficientnet_v2_l')()
    preproc_layer = effnet_pt.PreprocLayer()
    backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
    model = metrabs_pt.Metrabs(backbone, ji)
    model.eval()

    inp = torch.zeros((1, 3, cfg.proc_side, cfg.proc_side), dtype=torch.float32)
    intr = torch.eye(3, dtype=torch.float32)[np.newaxis]

    model((inp, intr))
    model.load_state_dict(torch.load(f'{model_dir}/ckpt.pt'))
    return model


def get_video(source, temppath='/tmp/video.mp4'):
    if not source.startswith('http'):
        return source

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(source, temppath)
    return temppath

if __name__ == '__main__':
    skeleton = 'smplx_42'
    get_config(f'{model_dir}/config.yaml')

    multiperson_model_pt = load_multiperson_model().cuda()
    source_dir = ''
    target_dir = ''

    for root, dirs, files in os.walk(source_dir):
        for file in tqdm(files, desc="Extracting Poses from NTU Dataset"):
            if file.endswith('.avi'):  
                video_path = os.path.join(root, file)
                
                # Construct the target path
                relative_path = os.path.relpath(root, source_dir)
                target_folder = os.path.join(target_dir, relative_path)
                target_file_path = os.path.join(target_folder, f"{os.path.splitext(file)[0]}.pt")
                
                # Skip if poses have already been extracted
                if os.path.exists(target_file_path):
                    continue

                poses = extract_poses(video_path, multiperson_model_pt, skeleton) 
                os.makedirs(target_folder, exist_ok=True) 
                # Save poses
                torch.save(poses, target_file_path)


