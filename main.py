import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep

import torch
import math

import torch.optim as optim

# Add device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_experiment_name(model_name, dataset, args):
    """生成实验名称，优先使用用户指定的名称"""
    if args.experiment_name:
        return args.experiment_name
    
    # 如果没有指定实验名称，则根据模型和数据集生成
    base_name = f"{model_name}_{dataset}"
    
    # 如果是TranAD_QC，添加消融研究参数
    if 'TranAD_QC' in model_name:
        ablation_parts = []
        if hasattr(args, 'activation_type') and args.activation_type != 'learnable_cosine':
            ablation_parts.append(args.activation_type)
        if hasattr(args, 'loss_type') and args.loss_type != 'complex_aware':
            ablation_parts.append(args.loss_type)
        if hasattr(args, 'focus_type') and args.focus_type != 'complex_aware':
            ablation_parts.append(args.focus_type)
        
        if ablation_parts:
            base_name += "_" + "_".join(ablation_parts)
    
    return base_name

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		elsoke: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def convert_to_windows_bounded(data, model, file_ids = None, label_rows = None):
    """窗口化且不跨越边界：
    - file_ids: 与 data 同长度的向量，标识每行所属文件；边界处窗口将以段首元素填充，不跨文件。
    - label_rows: 可选，与 data 同长度的 0/1 行标签；若提供，将在 0/1 变化处断开，避免窗口跨越异常转折。
    """
    w_size = model.n_window
    N = data.shape[0]
    # 计算每个位置的 segment_start（最近的边界之后的起点）
    if file_ids is None and label_rows is None:
        return convert_to_windows(data, model)
    # 边界标记：文件变化或标签从 0/1 发生变化
    boundary = torch.zeros(N, dtype=torch.bool, device=data.device)
    if file_ids is not None:
        if not torch.is_tensor(file_ids):
            file_ids = torch.from_numpy(file_ids).to(data.device)
        boundary[1:] |= (file_ids[1:] != file_ids[:-1])
    if label_rows is not None:
        if not torch.is_tensor(label_rows):
            label_rows = torch.from_numpy(label_rows.astype(bool)).to(data.device)
        boundary[1:] |= (label_rows[1:] != label_rows[:-1])
    # 计算每个 i 的当前段起点 idx_start
    idx_start = torch.zeros(N, dtype=torch.long, device=data.device)
    last = 0
    for i in range(N):
        if i == 0 or boundary[i]:
            last = i
        idx_start[i] = last
    # 组装窗口，段内不足则用段首元素填充
    windows = []
    for i in range(N):
        start = max(idx_start[i], i - w_size)
        seg_head = idx_start[i]
        needed = w_size - (i - start)
        if needed > 0:
            pad = data[seg_head:seg_head+1].repeat(needed, 1)
            w = torch.cat([pad, data[start:i]], dim=0)
        else:
            w = data[i-w_size:i]
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    return torch.stack(windows)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list, experiment_name=None):
	if experiment_name is None:
		experiment_name = get_experiment_name(model.name, args.dataset, args)
	
	folder = f'checkpoints/{experiment_name}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	
	# For TranAD_QC, pass ablation study parameters
	if modelname == 'TranAD_QC':
		model = model_class(dims, 
							activation_type=args.activation_type,
							loss_type=args.loss_type, 
							focus_type=args.focus_type).double().to(device)
	else:
		model = model_class(dims).double().to(device)  # Move model to GPU
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	# 生成实验名称
	experiment_name = get_experiment_name(modelname, args.dataset, args)
	# fname = f'checkpoints/{experiment_name}/model.ckpt'
	fname = f'/home/xiaoma/jiedan/TranAD-main/TranAD-main/checkpoints/TranAD_QC_my_data/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname, map_location=device)  # Load checkpoint to GPU
		
		# 过滤掉分类头相关的权重，避免维度不匹配
		model_state_dict = checkpoint['model_state_dict']
		filtered_state_dict = {}
		
		classifier_keys = []
		for key in model_state_dict.keys():
			if 'classifier' in key:
				classifier_keys.append(key)
			else:
				filtered_state_dict[key] = model_state_dict[key]
		
		if classifier_keys:
			print(f"跳过分类头权重加载 ({len(classifier_keys)} 个参数): 新架构需要重新训练{color.ENDC}")
			for key in classifier_keys[:3]:  # 只显示前3个
				print(f"  跳过: {key}")
			if len(classifier_keys) > 3:
				print(f"  ... 和其他 {len(classifier_keys)-3} 个分类头参数")
		
		# 允许缺失/新增键（例如新增 classifier 或架构改变）
		missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
		
		if missing_keys:
			print(f"新增参数 ({len(missing_keys)} 个): 将使用随机初始化{color.ENDC}")
			for key in missing_keys[:3]:  # 只显示前3个
				print(f"  新增: {key}")
			if len(missing_keys) > 3:
				print(f"  ... 和其他 {len(missing_keys)-3} 个新参数")
		
		if unexpected_keys:
			print(f"废弃参数 ({len(unexpected_keys)} 个): 已忽略{color.ENDC}")
		
		# 优化器和调度器的状态可能也不兼容，需要谨慎加载
		try:
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		except Exception as e:
			print(f"优化器/调度器状态加载失败，使用默认设置: {e}{color.ENDC}")
		
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def _train_tranad_c_adversarial(model, dataloader, optimizer, scheduler, epoch, l, feats):
	"""TranAD_C的完整对抗训练逻辑，按论文算法1实现"""
	n = epoch + 1
	# 使用更温和的epsilon参数，减少第一轮训练的剧烈变化
	epsilon = 1.005  # 从1.01改为1.005，让权重变化更平缓
	l1s, l2s = [], []
	
	# 创建分组优化器
	encoder_params = model.get_encoder_params()
	decoder1_params = model.get_decoder1_params()
	decoder2_params = model.get_decoder2_params()
	
	# 为新模型添加warmup：第一轮训练使用较小的学习率
	is_first_epoch = epoch == 0
	warmup_factor = 0.1 if is_first_epoch else 1.0
	
	if is_first_epoch:
		print(f"[Warmup] 第一轮训练使用{warmup_factor}倍学习率")
		for param_group in optimizer.param_groups:
			param_group['lr'] *= warmup_factor
	
	# MAML meta-learning相关
	meta_model = model.clone()
	meta_optimizer = torch.optim.AdamW(meta_model.parameters(), lr=model.meta_lr)
	
	for d, _ in tqdm(dataloader, leave=False, desc=f"Epoch {epoch} [{model.name}]", position=1):
		local_bs = d.shape[0]
		window = d.permute(1, 0, 2)
		elem = window[-1, :, :].view(1, local_bs, feats)
		
		# Phase 1: 获取O1, O2, Ô2
		o1, o2, o2_hat = model(window, elem, return_all=True)
		
		# 计算演化损失权重
		eps_weight = epsilon ** (-n)
		
		# L1 = ε^(-n) * ||O1 - W||² + (1 - ε^(-n)) * ||Ô2 - W||²
		l1_recon = l(o1, elem)
		l1_adv = l(o2_hat, elem)
		L1 = eps_weight * torch.mean(l1_recon) + (1 - eps_weight) * torch.mean(l1_adv)
		l1s.append(L1.detach().item())
		# L2 = ε^(-n) * ||O2 - W||² - (1 - ε^(-n)) * ||Ô2 - W||²
		# l2_recon = l(o2, elem)
		# L2 = eps_weight * torch.mean(l2_recon) - (1 - eps_weight) * torch.mean(l1_adv)
		
		# 分步优化：
		# Step A: 仅更新 编码器 + 解码器1 (minimize L1)
		optimizer.zero_grad(set_to_none=True)
		L1.backward()
		# 清除解码器2上的梯度，避免被误更新
		for param in decoder2_params:
			param.grad = None
		# 对本次要更新的分支做梯度裁剪
		torch.nn.utils.clip_grad_norm_(encoder_params + decoder1_params, 5.0)  # 放宽梯度范数限制
		# 移除过激进的逐元素梯度裁剪
		optimizer.step()

		# Step B: 仅更新 解码器2 (minimize L2)，需重新前向构图，避免在两次反向之间修改参数
		optimizer.zero_grad(set_to_none=True)
		# 重新计算前向输出与 L2
		o1_b, o2_b, o2_hat_b = model(window, elem, return_all=True)
		l2_recon_b = l(o2_b, elem)
		l1_adv_b = l(o2_hat_b, elem)
		L2_b = eps_weight * torch.mean(l2_recon_b) - (1 - eps_weight) * torch.mean(l1_adv_b)
		l2s.append(L2_b.detach().item())
		L2_b.backward()
		# 清除编码器与解码器1的梯度，避免被误更新
		for param in encoder_params + decoder1_params:
			param.grad = None
		# 对解码器2做梯度裁剪
		torch.nn.utils.clip_grad_norm_(decoder2_params, 5.0)  # 放宽梯度范数限制
		# 移除过激进的逐元素梯度裁剪
		optimizer.step()
		# 用最新的 L2 参与日志
		# L2 = L2_b
		
		# MAML meta-learning step
		# if n % 5 == 0:  # 每5个epoch做一次meta-learning
		# 	_maml_meta_update(model, meta_model, meta_optimizer, window, elem, l)

	scheduler.step()
	
	# 恢复学习率（如果使用了warmup）
	if is_first_epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] /= warmup_factor
		print(f"[Warmup] 学习率已恢复")
	
	tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s):.8f},\tL2 = {np.mean(l2s):.8f},\teps_weight = {eps_weight:.4f}')
	return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']

def _test_tranad_c(model, dataloader, l, bs, feats):
    alpha = 0.7  # 打分混合权重，仅用于 TranAD_C 测试
    all_scores = []
    all_o2hats = []
    
    with torch.no_grad():
        for d, _ in dataloader:
            # move batch to device
            d = d.to(device=device, non_blocking=True)
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, d.shape[0], feats)
            # 需要同时拿到 o1 和 o2_hat
            out = model(window, elem, return_all=True)
            # 兼容不同返回形式
            if isinstance(out, tuple):
                if len(out) == 3:
                    o1, _, o2_hat = out
                elif len(out) >= 2:
                    o1, o2_hat = out[0], out[1]
                else:
                    # 回退到原逻辑
                    z = out[0]
                    loss = l(z, elem)[0]
                    return loss.detach().cpu().numpy(), z.detach().cpu().numpy()[0]
            else:
                # 回退到仅有一个输出的情况
                z = out
                loss = l(z, elem)[0]
                return loss.detach().cpu().numpy(), z.detach().cpu().numpy()[0]

            # 计算混合打分：score = α*MSE(O1,W) + (1-α)*MSE(Ô2,W)
            mse_o1 = l(o1, elem)
            mse_o2hat = l(o2_hat, elem)
            # 数值健壮性处理，避免 NaN/Inf 造成后续阈值异常
            mse_o1 = torch.nan_to_num(mse_o1, nan=0.0, posinf=1e6, neginf=0.0)
            mse_o2hat = torch.nan_to_num(mse_o2hat, nan=0.0, posinf=1e6, neginf=0.0)
            score = alpha * mse_o1 + (1 - alpha) * mse_o2hat
            
            # 收集每个batch的结果，立即移到CPU避免GPU内存累积
            all_scores.append(score[0].detach().cpu())
            all_o2hats.append(o2_hat[0].detach().cpu())
            
            # 清理GPU内存
            del window, elem, out, o1, o2_hat, mse_o1, mse_o2hat, score
    
    # 合并所有batch的结果
    final_score = torch.cat(all_scores, dim=0)
    final_o2hat = torch.cat(all_o2hats, dim=0)
    
    return final_score.numpy(), final_o2hat.numpy()

def _maml_meta_update(model, meta_model, meta_optimizer, window, elem, l):
	"""MAML元学习更新，按论文Eq.(11)(12)实现"""
	# Inner loop: 在当前batch上做一步梯度更新
	meta_model.load_state_dict(model.state_dict())
	
	# 计算当前任务的损失
	o1, o2_hat = meta_model(window, elem)
	task_loss = torch.mean(l(o2_hat, elem))
	
	# Inner gradient update (θ' ← θ - α∇θL(f(θ)))
	inner_grads = torch.autograd.grad(task_loss, meta_model.parameters(), create_graph=True)
	
	# 手动更新参数
	updated_params = []
	for param, grad in zip(meta_model.parameters(), inner_grads):
		updated_params.append(param - model.meta_lr * grad)
	
	# Meta gradient update (θ ← θ - β∇θL(f(θ')))
	# 这里简化为直接用更新后的损失反向传播到原模型
	meta_optimizer.zero_grad()
	o1_updated, o2_hat_updated = meta_model(window, elem)
	meta_loss = torch.mean(l(o2_hat_updated, elem))
	meta_loss.backward()
	meta_optimizer.step()
	
	# 将meta学习的结果应用回原模型
	model.load_state_dict(meta_model.state_dict())

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in tqdm(data, leave=False, desc=f"Epoch {epoch} [{model.name}]", position=1):
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in tqdm(data, leave=False, desc=f"Epoch {epoch} [{model.name}]", position=1):
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(tqdm(data, leave=False, desc=f"Epoch {epoch} [{model.name}]", position=1)):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d in tqdm(data, leave=False, desc=f"Epoch {epoch} [{model.name}]", position=1):
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif model.name in ['MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(tqdm(data, leave=False, desc=f"Epoch {epoch} [{model.name}]", position=1)):
				x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data: 
				x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1; w_size = model.n_window
		mses, gls, dls = [], [], []
		if training:
			for d in tqdm(data, leave=False, desc=f"Epoch {epoch} [{model.name}]", position=1):
				# training discriminator
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# training generator
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
				# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'TranAD' in model.name:
		# Use configurable loss for TranAD_QC based on ablation parameters
		if 'TranAD_QC' in model.name:
			from src.models import get_loss_function
			l = get_loss_function(model.loss_type, alpha=0.9)
		else:
			l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data).to(device); dataset = TensorDataset(data_x, data_x)
		print(data_x[:3])
		bs = model.batch
		if training:
			dataloader = DataLoader(dataset, batch_size = bs, shuffle = True)
		else:
			dataloader = DataLoader(dataset, batch_size = bs, shuffle = False)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		
		# TranAD_C: 完整对抗训练逻辑
		if ('TranAD_C' in model.name or 'TranAD_QC' in model.name) and training:
			return _train_tranad_c_adversarial(model, dataloader, optimizer, scheduler, epoch, l, feats)
		elif ('TranAD_C' in model.name or 'TranAD_QC' in model.name) and not training:
			return _test_tranad_c(model, dataloader, l, bs, feats)
		# 原始TranAD逻辑
		elif training:
			for d, _ in tqdm(dataloader, leave=False, desc=f"Epoch {epoch} [{model.name}]", position=1):
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

def train_tranad_c_classifier(model, train_windows, train_labels, epochs, lr):
	"""在 TranAD_C 训练完成后，使用误差特征训练一个轻量分类头。
	- 输入特征：e = (o2_hat - tgt)^2 的最后一时刻向量 [B, feats]
	- 标签：行级标签（若多维，则使用“是否任一维为异常”的 0/1）
	- 类不平衡：使用 pos_weight，按标签比例自动计算
	"""
	model.eval()
	device_local = next(model.parameters()).device
	dtype_local = next(model.parameters()).dtype
	# 构建误差特征
	with torch.no_grad():
		X_list = []
		Y_list = []
		bs = model.batch
		data_x = torch.tensor(train_windows, dtype=dtype_local, device=device_local)
		dataset = TensorDataset(data_x, data_x)
		dl = DataLoader(dataset, batch_size=bs, shuffle=False)
		for d, _ in dl:
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, d.shape[0], train_windows.shape[-1])
			err = model.extract_error_features(window, elem, mode='complex')  # [B, feats], dtype matches model
			X_list.append(err.detach().cpu())
		# 行级标签（多维并为任一维为 1 即 1）
		row_labels = (train_labels.sum(axis=1) > 0).astype(np.float64)
		Y_list.append(torch.from_numpy(row_labels))
	X = torch.cat(X_list, dim=0).to(dtype=dtype_local)
	Y = torch.cat(Y_list, dim=0).to(dtype=dtype_local)
	# 去除可能的长度错位
	N = min(X.shape[0], Y.shape[0])
	X = X[:N]
	Y = Y[:N]
	# 构建 DataLoader
	dataset_clf = TensorDataset(X, Y)
	# 平衡采样：每个 batch 正负各一半（不足则有放回采样）
	batch_size = 2048
	half = max(1, batch_size // 2)
	Y_cpu = Y.detach().cpu().numpy()
	idx_pos_all = np.where(Y_cpu > 0.5)[0]
	idx_neg_all = np.where(Y_cpu <= 0.5)[0]
	if idx_pos_all.size == 0 or idx_neg_all.size == 0:
		print("[CLF] Warning: one class is empty; falling back to unbalanced batches.")
		idx_pos_all = np.arange(Y.shape[0])
		idx_neg_all = np.arange(0)
	# 损失与优化器（处理类不平衡）
	pos = float(Y.sum().item())
	neg = float(N - pos)
	
	# 针对不同分类头版本调整学习率和批次大小
	clf_version = getattr(args, 'clf_version', 'v1')
	if clf_version == 'v2':
		# V2需要更大的学习率和更小的批次，因为模型更复杂
		base_lr = 0.003  # 基础学习率
		batch_size = 1024  # 稍微减小批次大小，增加更新频率
		warmup_epochs = 15  # 更长的warmup
		print(f"[CLF V2] 使用优化参数: base_lr={base_lr:.4f}, batch_size={batch_size}, warmup_epochs={warmup_epochs}")
	elif clf_version == 'v3':
		# V3模型最复杂，需要最谨慎的训练
		base_lr = 0.002
		batch_size = 1536
		warmup_epochs = 20
		print(f"[CLF V3] 使用优化参数: base_lr={base_lr:.4f}, batch_size={batch_size}, warmup_epochs={warmup_epochs}")
	else:
		# V1现在也增强了，需要更好的训练策略
		base_lr = 0.004
		batch_size = 1500
		warmup_epochs = 10
		print(f"[CLF V1] 使用优化参数: base_lr={base_lr:.4f}, batch_size={batch_size}, warmup_epochs={warmup_epochs}")
	
	# 使用标准二分类损失：对 sigmoid 概率做 BCE，不再使用 logits 版损失
	criterion = nn.BCELoss()
	# 使用更适合复杂模型的优化器设置
	optimizer = optim.AdamW(model.classifier.parameters(), lr=base_lr, weight_decay=1e-4, eps=1e-8)
	
	# 添加余弦学习率调度器（对分类任务更友好）
	total_steps = epochs * max(1, int(np.ceil(N / batch_size)))
	warmup_steps = warmup_epochs * max(1, int(np.ceil(N / batch_size)))
	
	scheduler = optim.lr_scheduler.OneCycleLR(
		optimizer, 
		max_lr=base_lr * 2,  # 峰值学习率是基础的2倍
		total_steps=total_steps,
		pct_start=warmup_steps / total_steps,  # warmup占比
		anneal_strategy='cos',
		cycle_momentum=True,
		base_momentum=0.85,
		max_momentum=0.95,
		div_factor=10,  # 初始学习率 = max_lr / div_factor
		final_div_factor=100  # 最终学习率 = max_lr / final_div_factor
	)
	
	print(f"[CLF] 训练策略: warmup={warmup_steps}步, total={total_steps}步, 峰值lr={base_lr*2:.4f}")
	
	model.classifier.train()
	for ep in range(epochs):
		losses = []
		# 以“可覆盖的步数”为准构造批次
		steps = max(1, int(np.ceil(N / batch_size)))
		rng = np.random.default_rng(1234 + ep)
		for _ in range(steps):
			if idx_pos_all.size > 0:
				idx_pos = rng.choice(idx_pos_all, size=half, replace=idx_pos_all.size < half)
				X_pos = X[idx_pos]
				y_pos = torch.ones(half, dtype=dtype_local, device=device_local)
			else:
				X_pos = X[:0]
				y_pos = torch.zeros(0, dtype=dtype_local, device=device_local)
			if idx_neg_all.size > 0:
				idx_neg = rng.choice(idx_neg_all, size=half, replace=idx_neg_all.size < half)
				X_neg = X[idx_neg]
				y_neg = torch.zeros(half, dtype=dtype_local, device=device_local)
			else:
				X_neg = X[:0]
				y_neg = torch.zeros(0, dtype=dtype_local, device=device_local)
			xb = torch.cat([X_pos.to(device_local, dtype=dtype_local), X_neg.to(device_local, dtype=dtype_local)], dim=0)
			yb = torch.cat([y_pos, y_neg], dim=0)
			# 打乱
			perm = rng.permutation(xb.shape[0])
			xb = xb[perm]
			yb = yb[perm]
			logits = model.classifier(xb)
			probs = torch.sigmoid(logits)
			loss = criterion(probs, yb)
			optimizer.zero_grad()
			loss.backward()
			
			# 梯度裁剪，防止梯度爆炸
			torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
			
			optimizer.step()
			# 每个batch后更新学习率（OneCycleLR需要）
			scheduler.step()
			
			losses.append(loss.item())
		epoch_loss = np.mean(losses)
		current_lr = optimizer.param_groups[0]['lr']
		print(f"[CLF] epoch {ep+1}/{epochs} loss={epoch_loss:.4f} pos={int(pos)}/{N} lr={current_lr:.6f}")
		
		# 使用学习率调度器 - OneCycleLR不需要手动step
		# scheduler.step(epoch_loss)
		
	model.classifier.eval()


def eval_with_classifier(model, test_windows, test_labels):
	"""使用已训练的分类头进行评估，返回二值预测与分数。"""
	device_local = next(model.parameters()).device
	dtype_local = next(model.parameters()).dtype
	bs = model.batch
	with torch.no_grad():
		data_x = torch.tensor(test_windows, dtype=dtype_local, device=device_local)
		dataset = TensorDataset(data_x, data_x)
		dl = DataLoader(dataset, batch_size=bs, shuffle=False)
		logits_list = []
		for d, _ in dl:
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, d.shape[0], test_windows.shape[-1])
			logits = model.predict_scores(window, elem,mode='complex')  # [B]
			logits_list.append(logits.detach().cpu())
		logits_all = torch.cat(logits_list, dim=0).numpy()
	# 将 logits 映射为概率并用 0.5 阈值二值化
	probs_all = 1.0 / (1.0 + np.exp(-logits_all))
	thr = 0.5
	pred = (probs_all >= thr).astype(np.float64)
	# 行级标签
	labels_row = (test_labels.sum(axis=1) > 0).astype(np.float64)
	f1, precision, recall, TP, TN, FP, FN, roc_auc = calc_point2point(pred, labels_row)
	return {
		'f1': f1, 'precision': precision, 'recall': recall,
		'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'ROC/AUC': roc_auc,
		'threshold': thr
	}, pred, probs_all

def debug_tranad_qc_model(model, sample_window, sample_elem):
	"""调试TranAD_QC模型的输出和特征提取"""
	print(f"\n{color.HEADER}=== TranAD_QC 模型调试信息 ==={color.ENDC}")
	
	with torch.no_grad():
		# 检查输入形状
		print(f"输入形状: window={sample_window.shape}, elem={sample_elem.shape}")
		
		# 检查forward输出
		o1, o2, o2_hat = model(sample_window, sample_elem, return_all=True)
		print(f"模型输出形状: o1={o1.shape}, o2={o2.shape}, o2_hat={o2_hat.shape}")
		
		# 检查输出的数值范围
		print(f"o1 范围: [{o1.min().item():.4f}, {o1.max().item():.4f}], mean={o1.mean().item():.4f}")
		print(f"o2 范围: [{o2.min().item():.4f}, {o2.max().item():.4f}], mean={o2.mean().item():.4f}")
		print(f"o2_hat 范围: [{o2_hat.min().item():.4f}, {o2_hat.max().item():.4f}], mean={o2_hat.mean().item():.4f}")
		
		# 检查是否有NaN或Inf
		print(f"o1 NaN: {torch.isnan(o1).sum().item()}, Inf: {torch.isinf(o1).sum().item()}")
		print(f"o2 NaN: {torch.isnan(o2).sum().item()}, Inf: {torch.isinf(o2).sum().item()}")
		print(f"o2_hat NaN: {torch.isnan(o2_hat).sum().item()}, Inf: {torch.isinf(o2_hat).sum().item()}")
		
		# 检查误差特征提取
		error_feat = model.extract_error_features(sample_window, sample_elem, mode='complex')
		print(f"误差特征形状: {error_feat.shape}")
		print(f"误差特征范围: [{error_feat.min().item():.4f}, {error_feat.max().item():.4f}], mean={error_feat.mean().item():.4f}")
		print(f"误差特征 NaN: {torch.isnan(error_feat).sum().item()}, Inf: {torch.isinf(error_feat).sum().item()}")
		
		# 检查分类器输出
		if hasattr(model, 'classifier'):
			logits = model.predict_scores(sample_window, sample_elem)
			print(f"分类器输出形状: {logits.shape}")
			print(f"分类器输出范围: [{logits.min().item():.4f}, {logits.max().item():.4f}], mean={logits.mean().item():.4f}")
			print(f"分类器输出 NaN: {torch.isnan(logits).sum().item()}, Inf: {torch.isinf(logits).sum().item()}")
	
	print(f"{color.HEADER}=== 调试信息结束 ==={color.ENDC}\n")

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD.to(device), testD.to(device)  # Move data to GPU
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'MAD_GAN'] or 'TranAD' in model.name:
		# 对 my_data，使用带边界的窗口化，避免跨文件/标签边界糅合上下文
		if args.dataset == 'my_data':
			folder = os.path.join(output_folder, 'my_data')
			train_fids = np.load(os.path.join(folder, 'train_file_ids.npy'))
			test_fids = np.load(os.path.join(folder, 'test_file_ids.npy'))
			# 行级标签（用于测试阶段避免跨异常边界）
			label_rows_test = (labels.sum(axis=1) > 0)
			trainD = convert_to_windows_bounded(trainD, model, file_ids=train_fids)
			testD = convert_to_windows_bounded(testD, model, file_ids=test_fids, label_rows=label_rows_test)
		else:
			trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
	# 训练阶段样本统计（窗口级：训练集全正常）
	print(f"[LOG][Train] windows total={trainD.shape[0]} normal={trainD.shape[0]} anomaly=0")
	# 测试集样本统计（行级）
	test_row_labels = (labels.sum(axis=1) > 0)
	print(f"[LOG][Data Prepared] test rows total={labels.shape[0]} normal={int((~test_row_labels).sum())} anomaly={int(test_row_labels.sum())}")

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		
		# 可配置的训练轮数，默认为较少的轮数避免过训练
		if hasattr(args, 'num_epochs') and args.num_epochs is not None:
			experiment_name = get_experiment_name(args.model, args.dataset, args)
			num_epochs = args.num_epochs
		else:
			experiment_name = get_experiment_name(args.model, args.dataset, args)
			num_epochs = 10 if os.path.exists(f'checkpoints/{experiment_name}/model.ckpt') else 20
		
		# 如果是加载的预训练模型，使用更小的学习率进行微调
		if os.path.exists(f'checkpoints/TranAD_QC_my_data/model.ckpt') and not args.retrain:
			print(f"检测到预训练模型，使用较小学习率进行微调{color.ENDC}")
			for param_group in optimizer.param_groups:
				param_group['lr'] *= 0.1  # 降低学习率到1/10
			num_epochs = min(num_epochs, 3)  # 限制微调轮数
			
		print(f"将训练 {num_epochs} 轮")
		e = epoch + 1; start = time()
		
		# 早停相关变量
		best_loss = float('inf')
		patience = 3
		no_improve_count = 0
		
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1)), position=0):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
			
			# 早停检查
			if lossT < best_loss:
				best_loss = lossT
				no_improve_count = 0
				# 保存最佳模型
				save_model(model, optimizer, scheduler, e, accuracy_list, experiment_name)
			else:
				no_improve_count += 1
				
			# 如果是预训练模型且损失开始上升，提前停止
			if (os.path.exists(f'checkpoints/{experiment_name}/model.ckpt') and 
				not args.retrain and no_improve_count >= patience):
				print(f"{color.YELLOW}损失连续{patience}轮未改善，提前停止训练{color.ENDC}")
				break
				
			# 检查损失是否异常
			if lossT > best_loss * 2 and e > epoch + 2:
				print(f"{color.RED}警告：损失异常增大，可能过拟合。当前:{lossT:.6f}, 最佳:{best_loss:.6f}{color.ENDC}")
			
			# Release memory
			torch.cuda.empty_cache()
			
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
		
		# 调试TranAD_QC模型
		# if model.name == 'TranAD_QC':
		# 	# 取一个样本进行调试
		# 	sample_window = trainD[:1].permute(1, 0, 2)  # [w, 1, feats]
		# 	sample_elem = sample_window[-1:, :, :]  # [1, 1, feats]
		# 	debug_tranad_qc_model(model, sample_window, sample_elem)
		
		# 训练分类头（可选，TranAD_C/TranAD_QC）
		if args.train_clf and ('TranAD_C' in model.name or 'TranAD_QC' in model.name):
			print(f"{color.HEADER}Training classifier head on error features{color.ENDC}")
			
			# 选项：冻结TranAD主干，只训练分类头
			if getattr(args, 'freeze_backbone', False):
				print(f"{color.YELLOW}冻结TranAD主干，只训练分类头{color.ENDC}")
				# 冻结除分类头外的所有参数
				for name, param in model.named_parameters():
					if 'classifier' not in name:
						param.requires_grad_(False)
				print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
			
			# 从预处理产物加载分类头训练集（包含异常与正常），避免信息泄漏且可复现
			folder = os.path.join(output_folder, 'my_data' if args.dataset == 'my_data' else args.dataset)
			clf_train = np.load(os.path.join(folder, 'clf_train.npy'))
			clf_labels = np.load(os.path.join(folder, 'clf_labels.npy'))
			clf_fids = np.load(os.path.join(folder, 'clf_file_ids.npy'))
			# 打印分类头训练集行级样本统计
			clf_row_labels = (clf_labels.sum(axis=1) > 0)
			print(f"[LOG][Clf Train] rows total={clf_labels.shape[0]} normal={int((~clf_row_labels).sum())} anomaly={int(clf_row_labels.sum())}")
			
			# 设置分类头训练参数
			if not getattr(args, 'freeze_backbone', False):
				# 如果没有冻结主干，冻结 TranAD 参数，仅训练分类头
				for p in model.parameters():
					p.requires_grad_(False)
				for p in model.classifier.parameters():
					p.requires_grad_(True)
			
			row_labels = (clf_labels.sum(axis=1) > 0)
			clf_windows = convert_to_windows_bounded(torch.from_numpy(clf_train), model, file_ids=clf_fids, label_rows=row_labels)
			train_tranad_c_classifier(model, clf_windows, clf_labels, epochs=100, lr=0.01)
			# 保存包含分类头的模型
			save_model(model, optimizer, scheduler, e, accuracy_list, experiment_name)

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	if args.use_clf and ('TranAD_C' in model.name or 'TranAD_QC' in model.name):
		# 使用分类头评估（打印测试集样本统计）
		labels_row = (labels.sum(axis=1) > 0)
		print(f"[LOG][Test(use_clf)] rows total={labels.shape[0]} normal={int((~labels_row).sum())} anomaly={int(labels_row.sum())}")
		res, pred_bin, logits = eval_with_classifier(model, testD, labels)
		print(pd.DataFrame([res]))
		pprint(res)
		
		# 生成可视化（如果启用）
		if args.visualize:
			print(f"\n{color.HEADER}Generating classification result visualizations...{color.ENDC}")
			try:
				from visualization import visualize_classification_results
				# 转换testD为numpy数组
				if isinstance(testD, torch.Tensor):
					test_windows_np = testD.detach().cpu().numpy()
				else:
					test_windows_np = testD
				
				# 生成实验名称
				experiment_name = get_experiment_name(model.name, args.dataset, args)
				
				visualize_classification_results(
					model=model,
					test_windows=test_windows_np,
					test_labels=labels,
					model_name=experiment_name,
					n_folds=args.vis_n_folds,
					save_dir=args.vis_save_dir
				)
			except Exception as e:
				print(f"{color.RED}Visualization generation failed: {e}{color.ENDC}")
				import traceback
				traceback.print_exc()
		
		# 直接返回以避免后续 POT
		exit(0)
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
	# Sanity checks and cleanup
	if isinstance(loss, torch.Tensor): loss = loss.detach().cpu().numpy()
	if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
	loss = np.asarray(loss)
	y_pred = np.asarray(y_pred)
	labels = np.asarray(labels)
	# replace non-finite values which can break POT
	loss = np.nan_to_num(loss, nan=0.0, posinf=np.nanmedian(loss[np.isfinite(loss)]) if np.any(np.isfinite(loss)) else 0.0, neginf=0.0)
	y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)
	# align lengths if any mismatch
	T = min(loss.shape[0], labels.shape[0])
	if loss.shape[0] != labels.shape[0]:
		print(f"Length mismatch: loss({loss.shape[0]}), labels({labels.shape[0]}). Truncating to {T}.")
		loss = loss[:T]
		labels = labels[:T]
	# 打印测试阶段样本统计（行级）
	labels_row2 = (labels.sum(axis=1) > 0)
	print(f"[LOG][Test] rows total={labels.shape[0]} normal={int((~labels_row2).sum())} anomaly={int(labels_row2.sum())}")

	### Plot curves
	# if not args.test:
	# 	# 将 testO 移到 CPU 并转为 numpy，再传给 plotter（避免 CUDA tensor 转 numpy 报错）
	# 	if torch.is_tensor(testO):
	# 		if 'TranAD' in model.name:
	# 			testO = torch.roll(testO, 1, 0)
	# 		testO_np = testO.detach().cpu()
	# 	else:
	# 		testO_np = testO
	# 	plotter(f'{args.model}_{args.dataset}', testO_np, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	preds = []
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	if isinstance(lossT, torch.Tensor): lossT = lossT.detach().cpu().numpy()
	lossT = np.asarray(lossT)
	lossT = np.nan_to_num(lossT, nan=0.0, posinf=np.nanmedian(lossT[np.isfinite(lossT)]) if np.any(np.isfinite(lossT)) else 0.0, neginf=0.0)
	# TT = min(loss.shape[0], loss.shape[0])
	# if lossT.shape[0] != loss.shape[0]:
	# 	print(f"Train/Test score length mismatch: train({lossT.shape[0]}), test({loss.shape[0]}). Truncating to {TT}.")
	# 	lossT = lossT[:TT]
	# 	loss = loss[:TT]
	# 	labels = labels[:TT]
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot_eval(lt, l, ls); preds.append(pred)
		df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
	# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
	# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	print(df)
	pprint(result)
	# pprint(getresults2(df, result))
	# beep(4)
