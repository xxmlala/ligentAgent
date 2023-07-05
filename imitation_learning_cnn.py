from agent import PolicyNet

import logging
import hydra
import torch
import numpy as np
# import ligent
import h5py
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from omegaconf import OmegaConf
from dotmap import DotMap
from hydra.utils import instantiate
from pyvirtualdisplay import Display
from tqdm import tqdm
import other_utils
import time
from env_dataset import EnvDataset
from agent.features import CLIPWrapper, Identity, SimpleCNNEncoder
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


logger = logging.getLogger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')
# device='cpu'
def eval_env(env, agent, episodes, seed, action_decoder, env_decoder):
    returns = []
    distance_info_s = []
    ligent.set_scenes_dir("./custom_scenes")
    # eval_start = time.time()
    # print("Start eval!", flush=True)
    for episode in range(episodes):
        # state, _ = env.reset(seed=np.random.randint(0, 10000) + seed)
        state_img, state_text = env.reset()
        env_decoder.reset()
        done, blocked = False, False
        while not (done or blocked):
            state_img = np.expand_dims(state_img, 0)
            state_text = np.expand_dims(state_text, 0)
            action = agent.get_action((state_img,state_text), sample=False)
            action_env = action_decoder.decode(action)
            # state, _, _, info = env.step(action_env)
            (state_img, state_text), _, _, info = env.step(**action_env)
            reward, done, blocked, cumulate_reward, elpased_step, distance_info = env_decoder.step(info)
        # returns.append(info['episode']['r'].item())
        returns.append(cumulate_reward)
        distance_info_s.append(distance_info)
    # print(f"eval costs {time.time()-eval_start} s!", flush=True)
    ligent.set_scenes_dir("")
    return np.mean(returns), np.std(returns), distance_info_s

@torch.no_grad()
def eval(model, data_loader, criterion, clip_encoder=None):
    model.eval()
    clip_encoder.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    for obs_Vs,obs_Ts, labels in tqdm(data_loader):
        # obs_T = torch.zeros((len(obs_Vs), 520), device=device)
        
        logits = model(obs_Vs, clip_encoder.encode_text(obs_Ts))
        labels = labels.type(torch.LongTensor).to(device)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        _, predicted = torch.max(logits,-1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = total_correct/total_samples
    average_loss = total_loss/len(data_loader)
    accuracy = balanced_accuracy_score(y_true=all_labels, y_pred=all_predictions)
    cm = confusion_matrix(y_true=all_labels, y_pred=all_predictions)
    return average_loss, accuracy, cm

def train(cfg, seed: int, log_dict: dict, logger: logging.Logger, train_loader, eval_loader):
    # env = ligent.Environment(path="/home/liuan/workspace/drl_project/ligent-linux-server/LIGENT.x86_64")
    # env_decoder = ComeHereEnv(distance_reward=10, success_reward=200, distance_min=1.2, step_penalty=1, episode_len=500, is_debug=True)
    # action_decoder = instantiate(cfg.action_decoder, device=device)
    other_utils.set_seed_everywhere("", seed)
    feature_net = instantiate(cfg.feature_net, device=device, text_encoder=Identity, img_encoder=SimpleCNNEncoder)
    agent = instantiate(cfg.agent, preprocess_net=feature_net, device=device)
    cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))

    model = PolicyNet(feature_net=agent.get_feature_net(), actor_net=agent.get_actor_net())
    # criterion = nn.CrossEntropyLoss(weight=torch.as_tensor([29.7,1.5,16.0,4.5], device=device))
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor([1.4,15.4,4.4], device=device))

    clip_encoder = CLIPWrapper(device)
    clip_prameters = clip_encoder.get_params()
    optimizer = optim.Adam(list(model.parameters()), lr=3e-4)
    # optimizer = optim.Adam(list(clip_prameters), lr=3e-4)
    # optimizer = optim.Adam(list(model.parameters()), lr=3e-4)
    best_eval_acc = 0
    not_ascending_epoch = 0
    epoch_cnt = 0
    while True:
        epoch_cnt += 1
        running_loss = 0.0
        model.train()
        clip_encoder.train()
        train_predictions, train_labels = [],[]
        for obs_V,obs_T, labels in tqdm(train_loader):
            optimizer.zero_grad()
            # obs_T = torch.zeros((len(obs_V), 520), device=device)
            outputs = model(obs_V, clip_encoder.encode_text(obs_T))
            _,t_predictions = torch.max(outputs,-1)
            train_predictions.extend(t_predictions.cpu().numpy())
            labels = labels.type(torch.LongTensor).to(device)
            train_labels.extend(labels.cpu().numpy())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # break
        train_accuracy = balanced_accuracy_score(y_true=train_labels, y_pred=train_predictions)
        eval_loss, eval_acc, c_matrix = eval(model, eval_loader, criterion=criterion, clip_encoder=clip_encoder)
        logger.info(f'epoch [{epoch_cnt}] training_loss: {running_loss / len(train_loader):.3f}, eval_loss: {eval_loss}, training_balanced_acc: {train_accuracy}, eval_acc: {eval_acc}')
        logger.info(f'confusion_matrix:\n{c_matrix}')
        running_loss = 0.0
        
        agent.save(f'epoch_{epoch_cnt}_')
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            not_ascending_epoch = 0
            agent.save(f'best_acc_')
        else:
            not_ascending_epoch += 1
        
        if not_ascending_epoch >= 5:
            break


def get_dataloader(f_path="/home/liuan/workspace/drl_project/ligentAgent/dataset/Cls3Episode1000.h5", batch_size=256):
    with h5py.File(f_path, 'r') as f:
        # Get the datasets
        obs_V_dataset = f['obs_V']
        obs_T_dataset = f['obs_T']
        action_dataset = f['action']
        
        # Convert the datasets to numpy arrays
        obs_V_dataset = obs_V_dataset[:]
        obs_T_dataset = obs_T_dataset[:]
        action_dataset = action_dataset[:]

    
    # Convert the numpy arrays to PyTorch Tensors
    # obs_V_tensor = torch.from_numpy(obs_V_dataset).to(device)
    # action_tensor = torch.from_numpy(action_dataset).to(device).type(torch.long)

    # random_idx = torch.randperm(action_tensor.size(0), device=device)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_index, eval_index = next(splitter.split(obs_V_dataset, action_dataset))
    train_envDataset = EnvDataset(obs_V=obs_V_dataset[train_index], obs_T=obs_T_dataset[train_index], action=action_dataset[train_index])
    eval_envDataset = EnvDataset(obs_V=obs_V_dataset[eval_index], obs_T=obs_T_dataset[eval_index], action=action_dataset[eval_index])
    # train_len = int(len(random_idx)*0.7)


    # Create a DataLoader from the TensorDataset
    train_dataloader = DataLoader(train_envDataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_envDataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, eval_dataloader


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    log_dict = other_utils.get_log_dict(cfg.agent._target_)
    for seed in cfg.seeds:
        with torch.autograd.set_detect_anomaly(True):
            train(cfg, seed, log_dict, logger, *(get_dataloader(batch_size=256)))
    


if __name__=="__main__":
    # with Display(visible=False) as disp:
    main()