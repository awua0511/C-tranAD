import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
# import dgl
# from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
import math
torch.manual_seed(1)

class ClfHead(nn.Module):
	def __init__(self, feats: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(feats, feats), nn.ReLU(True),
			nn.Dropout(0.2),
			nn.Linear(feats, 1)
		)
	def forward(self, x):
		return self.net(x).squeeze(-1)

class ComplexClfHeadV2(nn.Module):
	"""增强版复数分类头：多尺度特征提取 + 复数感知池化 + 残差连接"""
	def __init__(self, feats: int, hidden_complex: int = None, dropout_p: float = 0.2):
		super().__init__()
		assert feats % 2 == 0, 'ComplexClfHead expects even feature size (adjacent real/imag).'
		self.n_complex = feats // 2
		# 大幅增加隐藏维度，提供更强的非线性表达能力
		self.hidden_complex = hidden_complex if hidden_complex is not None else max(64, self.n_complex * 4)
		
		# 输入投影层，将输入扩展到更大维度
		self.input_proj = ComplexLinear(self.n_complex, self.hidden_complex)
		self.input_norm = ComplexLayerNorm(self.hidden_complex)
		
		# 多尺度特征提取分支 - 使用不同的激活参数
		self.branch1 = nn.ModuleDict({
			'lin': ComplexLinear(self.hidden_complex, self.hidden_complex),
			'act': ComplexCosineActivation(omega_init=0.5, phi_init=0.0),  # 更温和的激活
			'norm': ComplexLayerNorm(self.hidden_complex)
		})
		
		self.branch2 = nn.ModuleDict({
			'lin': ComplexLinear(self.hidden_complex, self.hidden_complex),
			'act': ComplexCosineActivation(omega_init=1.5, phi_init=np.pi/6),  # 中等强度
			'norm': ComplexLayerNorm(self.hidden_complex)
		})
		
		self.branch3 = nn.ModuleDict({
			'lin': ComplexLinear(self.hidden_complex, self.hidden_complex),
			'act': ComplexCosineActivation(omega_init=2.5, phi_init=np.pi/3),  # 更强的非线性
			'norm': ComplexLayerNorm(self.hidden_complex)
		})
		
		# 特征融合 - 不降维，保持表达能力
		self.fusion = ComplexLinear(self.hidden_complex * 3, self.hidden_complex * 2)  # 增加融合后的维度
		self.fusion_act = ComplexCosineActivation(omega_init=1.0, phi_init=0.0)
		self.fusion_norm = ComplexLayerNorm(self.hidden_complex * 2)
		
		# 复数感知特征分支 - 增加维度
		self.magnitude_branch = nn.Sequential(
			nn.Linear(self.hidden_complex * 2, self.hidden_complex),
			nn.ReLU(),
			nn.Dropout(dropout_p),
			nn.Linear(self.hidden_complex, self.hidden_complex // 2)
		)
		
		self.phase_branch = nn.Sequential(
			nn.Linear(self.hidden_complex * 2, self.hidden_complex),
			nn.ReLU(), 
			nn.Dropout(dropout_p),
			nn.Linear(self.hidden_complex, self.hidden_complex // 2)
		)
		
		# 原始复数特征分支
		self.complex_branch = nn.Sequential(
			nn.Linear(self.hidden_complex * 4, self.hidden_complex * 2),  # 输入是real+imag concat
			nn.ReLU(),
			nn.Dropout(dropout_p),
			nn.Linear(self.hidden_complex * 2, self.hidden_complex)
		)
		
		# 最终分类器 - 更深、更宽
		final_input_dim = self.hidden_complex + self.hidden_complex // 2 + self.hidden_complex // 2  # complex + mag + phase
		self.final_classifier = nn.Sequential(
			nn.Linear(final_input_dim, self.hidden_complex * 2),
			nn.ReLU(),
			nn.BatchNorm1d(self.hidden_complex * 2),
			nn.Dropout(dropout_p),
			nn.Linear(self.hidden_complex * 2, self.hidden_complex),
			nn.ReLU(),
			nn.Dropout(dropout_p),
			nn.Linear(self.hidden_complex, self.hidden_complex // 2),
			nn.ReLU(),
			nn.Linear(self.hidden_complex // 2, 1)
		)
		
	def forward(self, x):
		# x: [B, 2*C] (or [..., 2*C])
		real, imag = _split_complex_adjacent(x)
		
		# 输入投影到更高维度
		h_r, h_i = self.input_proj(real, imag)
		h_r, h_i = self.input_norm(h_r, h_i)
		
		# 多尺度特征提取 + 残差连接
		# Branch 1
		h1_r, h1_i = self.branch1['lin'](h_r, h_i)
		h1_r, h1_i = self.branch1['act'](h1_r, h1_i)
		h1_r, h1_i = self.branch1['norm'](h1_r, h1_i)
		h1_r = h1_r + h_r  # 残差连接
		h1_i = h1_i + h_i
		
		# Branch 2
		h2_r, h2_i = self.branch2['lin'](h_r, h_i)
		h2_r, h2_i = self.branch2['act'](h2_r, h2_i)
		h2_r, h2_i = self.branch2['norm'](h2_r, h2_i)
		h2_r = h2_r + h_r  # 残差连接
		h2_i = h2_i + h_i
		
		# Branch 3
		h3_r, h3_i = self.branch3['lin'](h_r, h_i)
		h3_r, h3_i = self.branch3['act'](h3_r, h3_i)
		h3_r, h3_i = self.branch3['norm'](h3_r, h3_i)
		h3_r = h3_r + h_r  # 残差连接
		h3_i = h3_i + h_i
		
		# 特征融合
		fused_r = torch.cat([h1_r, h2_r, h3_r], dim=-1)
		fused_i = torch.cat([h1_i, h2_i, h3_i], dim=-1)
		fused_r, fused_i = self.fusion(fused_r, fused_i)
		fused_r, fused_i = self.fusion_act(fused_r, fused_i)
		fused_r, fused_i = self.fusion_norm(fused_r, fused_i)
		
		# 复数感知特征：分别处理幅度、相位和原始复数特征
		# 1. 幅度特征
		magnitude = torch.sqrt(fused_r * fused_r + fused_i * fused_i + 1e-12)
		mag_feat = self.magnitude_branch(magnitude)
		
		# 2. 相位特征
		phase = torch.atan2(fused_i + 1e-12, fused_r + 1e-12)
		phase_feat = self.phase_branch(phase)
		
		# 3. 原始复数特征
		complex_feat_raw = torch.cat([fused_r, fused_i], dim=-1)
		complex_feat = self.complex_branch(complex_feat_raw)
		
		# 融合所有特征
		combined_feat = torch.cat([complex_feat, mag_feat, phase_feat], dim=-1)
		
		# 最终分类
		logits = self.final_classifier(combined_feat)
		return logits.squeeze(-1)

class ComplexClfHead(nn.Module):
	"""Complex-valued classifier head operating on interleaved real/imag features.
	Input: x[..., 2*C] laid out as (r0, i0, r1, i1, ...). Outputs real logit [...].
	Enhanced version with larger capacity.
	"""
	def __init__(self, feats: int, hidden_complex: int = None, dropout_p: float = 0.2):
		super().__init__()
		assert feats % 2 == 0, 'ComplexClfHead expects even feature size (adjacent real/imag).'
		self.n_complex = feats // 2
		# 大幅增加V1的隐藏维度，提供更强的表达能力
		self.hidden_complex = hidden_complex if hidden_complex is not None else max(32, self.n_complex * 2)
		
		# 输入投影层
		self.input_proj = ComplexLinear(self.n_complex, self.hidden_complex)
		self.input_norm = ComplexLayerNorm(self.hidden_complex)
		
		# 主要特征提取
		self.lin1 = ComplexLinear(self.hidden_complex, self.hidden_complex)
		self.act1 = ComplexCosineActivation(omega_init=1.0, phi_init=0.0)
		self.drop1 = ComplexDropout(dropout_p)
		self.norm1 = ComplexLayerNorm(self.hidden_complex)
		
		# 第二层特征提取
		self.lin2 = ComplexLinear(self.hidden_complex, self.hidden_complex // 2)
		self.act2 = ComplexCosineActivation(omega_init=1.5, phi_init=np.pi/6)
		self.drop2 = ComplexDropout(dropout_p)
		self.norm2 = ComplexLayerNorm(self.hidden_complex // 2)
		
		# 输出层 - 转换为实数分类器
		final_dim = self.hidden_complex // 2
		self.final_classifier = nn.Sequential(
			nn.Linear(final_dim * 2, final_dim),  # *2 because of real+imag concat
			nn.ReLU(),
			nn.Dropout(dropout_p),
			nn.Linear(final_dim, final_dim // 2),
			nn.ReLU(),
			nn.Linear(final_dim // 2, 1)
		)
		
	def forward(self, x):
		# x: [B, 2*C] (or [..., 2*C])
		real, imag = _split_complex_adjacent(x)
		
		# 输入投影
		h_r, h_i = self.input_proj(real, imag)
		h_r, h_i = self.input_norm(h_r, h_i)
		
		# 第一层 + 残差连接
		h1_r, h1_i = self.lin1(h_r, h_i)
		h1_r, h1_i = self.act1(h1_r, h1_i)
		h1_r, h1_i = self.drop1(h1_r, h1_i)
		h1_r, h1_i = self.norm1(h1_r, h1_i)
		h1_r = h1_r + h_r  # 残差连接
		h1_i = h1_i + h_i
		
		# 第二层
		h2_r, h2_i = self.lin2(h1_r, h1_i)
		h2_r, h2_i = self.act2(h2_r, h2_i)
		h2_r, h2_i = self.drop2(h2_r, h2_i)
		h2_r, h2_i = self.norm2(h2_r, h2_i)
		
		# 转换为实数特征并分类
		combined_features = torch.cat([h2_r, h2_i], dim=-1)
		logits = self.final_classifier(combined_features)
		return logits.squeeze(-1)

## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
	def __init__(self, feats):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 1
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

	def forward(self, x):
		hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64), 
			torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
		outputs = []
		for i, g in enumerate(x):
			multivariate_output = []
			for j in range(self.n_feats):
				univariate_input = g.view(-1)[j].view(1, 1, -1)
				out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
				multivariate_output.append(2 * out.view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)

## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = 5 # MHA w_size = 5
		self.n = self.n_feats * self.n_window
		self.atts = [ nn.Sequential( nn.Linear(self.n, feats * feats), 
				nn.ReLU(True))	for i in range(1)]
		self.atts = nn.ModuleList(self.atts)

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)		
		return g, ats

## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.lstm2 = nn.LSTM(feats, self.n_feats)
		self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
			out = self.fcn(out.view(-1))
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)

## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
	def __init__(self, feats):
		super(DAGMM, self).__init__()
		self.name = 'DAGMM'
		self.lr = 0.0001
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 8
		self.n_window = 5 # DAGMM w_size = 5
		self.n = self.n_feats * self.n_window
		self.n_gmm = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.estimate = nn.Sequential(
			nn.Linear(self.n_latent+2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
			nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
		)

	def compute_reconstruction(self, x, x_hat):
		relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
		cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
		return relative_euclidean_distance, cosine_similarity

	def forward(self, x):
		## Encode Decoder
		x = x.view(1, -1)
		z_c = self.encoder(x)
		x_hat = self.decoder(z_c)
		## Compute Reconstructoin
		rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
		z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
		## Estimate
		gamma = self.estimate(z)
		return z_c, x_hat.view(-1), z, gamma.view(-1)

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
	def __init__(self, feats):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = 0.002
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden

## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 5
		self.n_window = 5 # USAD w_size = 5
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = self.encoder(g.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)

## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
	def __init__(self, feats):
		super(MSCRED, self).__init__()
		self.name = 'MSCRED'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = feats
		self.encoder = nn.ModuleList([
			ConvLSTM(1, 32, (3, 3), 1, True, True, False),
			ConvLSTM(32, 64, (3, 3), 1, True, True, False),
			ConvLSTM(64, 128, (3, 3), 1, True, True, False),
			]
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.n_window)
		for cell in self.encoder:
			_, z = cell(z.view(1, *z.shape))
			z = z[0][0]
		## Decode
		x = self.decoder(z)
		return x.view(-1)

## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
	def __init__(self, feats):
		super(CAE_M, self).__init__()
		self.name = 'CAE_M'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = feats
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.n_window)
		z = self.encoder(z)
		## Decode
		x = self.decoder(z)
		return x.view(-1)

## MTAD_GAT Model (ICDM 20) - 需要DGL库，暂时注释
# class MTAD_GAT(nn.Module):
# 	def __init__(self, feats):
# 		super(MTAD_GAT, self).__init__()
# 		self.name = 'MTAD_GAT'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_window = feats
# 		self.n_hidden = feats * feats
# 		self.g = dgl.graph((torch.tensor(list(range(1, feats+1))), torch.tensor([0]*feats)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(feats, 1, feats)
# 		self.time_gat = GATConv(feats, 1, feats)
# 		self.gru = nn.GRU((feats+1)*feats*3, feats*feats, 1)

# 	def forward(self, data, hidden):
# 		hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
# 		data = data.view(self.n_window, self.n_feats)
# 		data_r = torch.cat((torch.zeros(1, self.n_feats), data))
# 		feat_r = self.feature_gat(self.g, data_r)
# 		data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
# 		time_r = self.time_gat(self.g, data_t)
# 		data = torch.cat((torch.zeros(1, self.n_feats), data))
# 		data = data.view(self.n_window+1, self.n_feats, 1)
# 		x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
# 		x, h = self.gru(x, hidden)
# 		return x.view(-1), h

## GDN Model (AAAI 21) - 需要DGL库，暂时注释
# class GDN(nn.Module):
# 	def __init__(self, feats):
# 		super(GDN, self).__init__()
# 		self.name = 'GDN'
# 		self.lr = 0.0001
# 		self.n_feats = feats
# 		self.n_window = 5
# 		self.n_hidden = 16
# 		self.n = self.n_window * self.n_feats
# 		src_ids = np.repeat(np.array(list(range(feats))), feats)
# 		dst_ids = np.array(list(range(feats))*feats)
# 		self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
# 		self.g = dgl.add_self_loop(self.g)
# 		self.feature_gat = GATConv(1, 1, feats)
# 		self.attention = nn.Sequential(
# 			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
# 		)
# 		self.fcn = nn.Sequential(
# 			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
# 			nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
# 		)

# 	def forward(self, data):
# 		# Bahdanau style attention
# 		att_score = self.attention(data).view(self.n_window, 1)
# 		data = data.view(self.n_window, self.n_feats)
# 		data_r = torch.matmul(data.permute(1, 0), att_score)
# 		# GAT convolution on complete graph
# 		feat_r = self.feature_gat(self.g, data_r)
# 		feat_r = feat_r.view(self.n_feats, self.n_feats)
# 		# Pass through a FCN
# 		x = self.fcn(feat_r)
# 		return x.view(-1)

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
	def __init__(self, feats):
		super(MAD_GAN, self).__init__()
		self.name = 'MAD_GAN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_window = 5 # MAD_GAN w_size = 5
		self.n = self.n_feats * self.n_window
		self.generator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.discriminator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Generate
		z = self.generator(g.view(1,-1))
		## Discriminator
		real_score = self.discriminator(g.view(1,-1))
		fake_score = self.discriminator(z.view(1,-1))
		return z.view(-1), real_score.view(-1), fake_score.view(-1)

# Proposed Model (VLDB 22)
class TranAD_Basic(nn.Module):
	def __init__(self, feats):
		super(TranAD_Basic, self).__init__()
		self.name = 'TranAD_Basic'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = RealCosineActivation(omega_init=1.0, phi_init=0.0)

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD_Transformer(nn.Module):
	def __init__(self, feats):
		super(TranAD_Transformer, self).__init__()
		self.name = 'TranAD_Transformer'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_hidden = 8
		self.n_window = 10
		self.n = 2 * self.n_feats * self.n_window
		self.transformer_encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), RealCosineActivation(omega_init=1.0, phi_init=0.0),
			nn.Linear(self.n_hidden, self.n), RealCosineActivation(omega_init=1.0, phi_init=0.0))
		self.transformer_decoder1 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), RealCosineActivation(omega_init=1.0, phi_init=0.0),
			nn.Linear(self.n_hidden, 2 * feats), RealCosineActivation(omega_init=1.0, phi_init=0.0))
		self.transformer_decoder2 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), RealCosineActivation(omega_init=1.0, phi_init=0.0),
			nn.Linear(self.n_hidden, 2 * feats), RealCosineActivation(omega_init=1.0, phi_init=0.0))
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), RealCosineActivation(omega_init=1.0, phi_init=0.0))

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src.permute(1, 0, 2).flatten(start_dim=1)
		tgt = self.transformer_encoder(src)
		return tgt

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.transformer_decoder1(self.encode(src, c, tgt))
		x1 = x1.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x1 = self.fcn(x1)
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.transformer_decoder2(self.encode(src, c, tgt))
		x2 = x2.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x2 = self.fcn(x2)
		return x1, x2

# Proposed Model + Self Conditioning + MAML (VLDB 22)
class TranAD_Adversarial(nn.Module):
	def __init__(self, feats):
		super(TranAD_Adversarial, self).__init__()
		self.name = 'TranAD_Adversarial'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), RealCosineActivation(omega_init=1.0, phi_init=0.0))

	def encode_decode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x = self.encode_decode(src, c, tgt)
		# Phase 2 - With anomaly scores
		c = (x - src) ** 2
		x = self.encode_decode(src, c, tgt)
		return x

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD_SelfConditioning(nn.Module):
	def __init__(self, feats):
		super(TranAD_SelfConditioning, self).__init__()
		self.name = 'TranAD_SelfConditioning'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
	def __init__(self, feats):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2

# Complete TranAD with full adversarial training and MAML (复数网络友好设计)
class TranAD_C(nn.Module):
	def __init__(self, feats):
		super(TranAD_C, self).__init__()
		self.name = 'TranAD_C'
		self.lr = lr
		self.batch = 512
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		
		# Encoder (shared by both phases)
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		
		# Two separate decoders for adversarial training
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		
		# Output layers (便于后续改为复数)
		self.fcn1 = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
		self.fcn2 = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
		
		# MAML related parameters
		self.meta_lr = 0.001
		self.adaptation_steps = 1
		# Classification head (optional)
		self.classifier = ClfHead(self.n_feats)

	def encode(self, src, c, tgt):
		"""编码器部分，后续可方便改为复数运算"""
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward_phase1(self, src, tgt):
		"""Phase 1: 无异常分数的重构"""
		c = torch.zeros_like(src)
		tgt_enc, memory = self.encode(src, c, tgt)
		o1 = self.fcn1(self.transformer_decoder1(tgt_enc, memory))
		o2 = self.fcn2(self.transformer_decoder2(tgt_enc, memory))
		return o1, o2

	def forward_phase2(self, src, tgt, focus_score):
		"""Phase 2: 基于focus score的重构"""
		tgt_enc, memory = self.encode(src, focus_score, tgt)
		o2_hat = self.fcn2(self.transformer_decoder2(tgt_enc, memory))
		return o2_hat

	def forward(self, src, tgt, return_all=False):
		"""完整前向传播，支持两阶段"""
		# Phase 1
		o1, o2 = self.forward_phase1(src, tgt)
		
		# Phase 2 with focus score (real-valued: reconstruction error)
		focus_score = (o1 - src) ** 2
		o2_hat = self.forward_phase2(src, tgt, focus_score)
		
		if return_all:
			return o1, o2, o2_hat  # 返回所有输出用于对抗训练
		else:
			return o1, o2_hat  # 默认返回，与原TranAD兼容

	def clone(self):
		"""为MAML创建模型副本"""
		clone = TranAD_C(self.n_feats)
		clone.load_state_dict(self.state_dict())
		return clone

	def get_encoder_params(self):
		"""获取编码器参数（用于分组优化）"""
		params = []
		params.extend(self.transformer_encoder.parameters())
		params.extend(self.pos_encoder.parameters())
		return params

	def get_decoder1_params(self):
		"""获取解码器1参数"""
		params = []
		params.extend(self.transformer_decoder1.parameters())
		params.extend(self.fcn1.parameters())
		return params

	def get_decoder2_params(self):
		"""获取解码器2参数"""
		params = []
		params.extend(self.transformer_decoder2.parameters())
		params.extend(self.fcn2.parameters())
		return params
	
	def extract_error_features(self, src, tgt):
		"""返回单步误差特征 e = (o2_hat - tgt)^2 的最后一时刻特征向量 [B, feats]"""
		with torch.no_grad():
			out = self.forward(src, tgt, return_all=True)
			if isinstance(out, tuple) and len(out) == 3:
				_, _, o2_hat = out
			else:
				_, o2_hat = out
			# err = (o2_hat - tgt) ** 2  # [1, B, feats]
			return o2_hat[0]
	
	def predict_scores(self, src, tgt):
		"""用分类头得到每个样本的 logit 分数（无需 sigmoid），输出 [B]"""
		err = self.extract_error_features(src, tgt)  # [B, feats]
		logits = self.classifier(err)
		return logits

# ----------------- Complex-valued components and TranAD_QC -----------------

def _split_complex_adjacent(x):
	"""Split adjacent real/imag pairs along the last dim.
	Input: [..., 2*C]
	Returns: real[... , C], imag[... , C]
	"""
	real = x[..., 0::2]
	imag = x[..., 1::2]
	return real, imag


def _merge_complex_adjacent(real, imag):
	"""Interleave real/imag back to adjacent layout along last dim.
	Inputs: real[... , C], imag[... , C] -> out[... , 2*C] with (r0,i0,r1,i1,...)
	"""
	x = torch.stack((real, imag), dim=-1)
	return x.flatten(-2)


class ComplexLinear(nn.Module):
	def __init__(self, in_features: int, out_features: int, bias: bool = True):
		super().__init__()
		self.wr = nn.Linear(in_features, out_features, bias=bias)
		self.wi = nn.Linear(in_features, out_features, bias=bias)
	def forward(self, real, imag):
		# (Wr + iWi)(r + i i) = (Wr r - Wi i) + i (Wr i + Wi r)
		re = self.wr(real) - self.wi(imag)
		im = self.wr(imag) + self.wi(real)
		return re, im


class ComplexActivation(nn.Module):
	def __init__(self, act: nn.Module = None):
		super().__init__()
		self.act = act if act is not None else nn.LeakyReLU(True)
	def forward(self, real, imag):
		return self.act(real), self.act(imag)


class ComplexReLU(nn.Module):
	"""Complex ReLU activation: applies ReLU to both real and imaginary parts separately."""
	def __init__(self):
		super().__init__()
		self.relu = nn.ReLU()
	
	def forward(self, real, imag):
		return self.relu(real), self.relu(imag)


class ComplexLeakyReLU(nn.Module):
	"""Complex LeakyReLU activation: applies LeakyReLU to both real and imaginary parts separately."""
	def __init__(self, negative_slope: float = 0.01):
		super().__init__()
		self.leaky_relu = nn.LeakyReLU(negative_slope)
	
	def forward(self, real, imag):
		return self.leaky_relu(real), self.leaky_relu(imag)


class ComplexFixedCosineActivation(nn.Module):
	"""Non-learnable complex cosine activation with fixed parameters."""
	def __init__(self, omega: float = 1.0, phi: float = 0.0):
		super().__init__()
		# Fixed parameters (not learnable)
		self.register_buffer('omega', torch.tensor(omega, dtype=torch.float64))
		self.register_buffer('phi', torch.tensor(phi, dtype=torch.float64))
	
	def forward(self, real, imag):
		# 添加小常数避免除零和数值不稳定
		eps = 1e-12
		magnitude = torch.sqrt(real * real + imag * imag + eps)
		
		# 当幅度很小时，直接返回输入，避免atan2的数值问题
		small_magnitude_mask = magnitude < eps
		
		# Compute argument (phase angle) of complex number
		arg_z = torch.atan2(imag + eps, real + eps)
		
		# Compute activation: (1/2) * (1 + cos(ω * arg(z) + φ))
		activation_factor = 0.5 * (1.0 + torch.cos(self.omega * arg_z + self.phi))
		
		# 对于幅度很小的情况，使用恒等映射
		activation_factor = torch.where(small_magnitude_mask, torch.ones_like(activation_factor), activation_factor)
		
		# Apply activation to both real and imaginary parts
		real_out = activation_factor * real
		imag_out = activation_factor * imag
		
		return real_out, imag_out


class ComplexCosineActivation(nn.Module):
	"""Complex activation function: σ(z) = (1/2) * (1 + cos(ω * arg(z) + φ)) * z
	where ω and φ are learnable parameters.
	"""
	def __init__(self, omega_init: float = 1.0, phi_init: float = 0.0):
		super().__init__()
		# Learnable parameters ω and φ，使用double类型以匹配模型
		self.omega = nn.Parameter(torch.tensor(omega_init, dtype=torch.float64))
		self.phi = nn.Parameter(torch.tensor(phi_init, dtype=torch.float64))
	
	def forward(self, real, imag):
		# 添加小常数避免除零和数值不稳定
		eps = 1e-12
		magnitude = torch.sqrt(real * real + imag * imag + eps)
		
		# 当幅度很小时，直接返回输入，避免atan2的数值问题
		small_magnitude_mask = magnitude < eps
		
		# Compute argument (phase angle) of complex number
		arg_z = torch.atan2(imag + eps, real + eps)
		
		# Compute activation: (1/2) * (1 + cos(ω * arg(z) + φ))
		activation_factor = 0.5 * (1.0 + torch.cos(self.omega * arg_z + self.phi))
		
		# 对于幅度很小的情况，使用恒等映射
		activation_factor = torch.where(small_magnitude_mask, torch.ones_like(activation_factor), activation_factor)
		
		# Apply activation to both real and imaginary parts
		real_out = activation_factor * real
		imag_out = activation_factor * imag
		
		return real_out, imag_out


class ComplexSigmoid(nn.Module):
	"""Complex-valued Sigmoid activation function.
	Applies sigmoid to both real and imaginary parts separately.
	"""
	def __init__(self):
		super().__init__()
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, real, imag):
		real_out = self.sigmoid(real)
		imag_out = self.sigmoid(imag)
		return real_out, imag_out


class RealCosineActivation(nn.Module):
	"""Real-valued version of the cosine activation function for TranAD models.
	σ(x) = (1/2) * (1 + cos(ω * x + φ)) * x
	where ω and φ are learnable parameters.
	"""
	def __init__(self, omega_init: float = 1.0, phi_init: float = 0.0):
		super().__init__()
		# Learnable parameters ω and φ，使用double类型以匹配模型
		self.omega = nn.Parameter(torch.tensor(omega_init, dtype=torch.float64))
		self.phi = nn.Parameter(torch.tensor(phi_init, dtype=torch.float64))
	
	def forward(self, x):
		# Compute activation: (1/2) * (1 + cos(ω * x + φ)) * x
		activation_factor = 0.5 * (1.0 + torch.cos(self.omega * x + self.phi))
		return activation_factor * x


class ComplexDropout(nn.Module):
	def __init__(self, p: float = 0.0):
		super().__init__()
		self.drop = nn.Dropout(p)
	def forward(self, real, imag):
		return self.drop(real), self.drop(imag)


class ComplexLayerNorm(nn.Module):
	"""Complex-valued Layer Normalization.
	Applies layer normalization to both real and imaginary parts separately.
	"""
	def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
		super().__init__()
		# Ensure normalized_shape is a tuple
		if isinstance(normalized_shape, int):
			self.normalized_shape = (normalized_shape,)
		else:
			self.normalized_shape = normalized_shape
		self.eps = eps
		self.elementwise_affine = elementwise_affine
		if self.elementwise_affine:
			self.weight_r = nn.Parameter(torch.ones(self.normalized_shape))
			self.weight_i = nn.Parameter(torch.ones(self.normalized_shape))
			self.bias_r = nn.Parameter(torch.zeros(self.normalized_shape))
			self.bias_i = nn.Parameter(torch.zeros(self.normalized_shape))
		else:
			self.register_parameter('weight_r', None)
			self.register_parameter('weight_i', None)
			self.register_parameter('bias_r', None)
			self.register_parameter('bias_i', None)
	
	def forward(self, real, imag):
		# Apply layer norm to real and imaginary parts separately
		real_norm = F.layer_norm(real, self.normalized_shape, None, None, self.eps)
		imag_norm = F.layer_norm(imag, self.normalized_shape, None, None, self.eps)
		
		if self.elementwise_affine:
			real_out = self.weight_r * real_norm + self.bias_r
			imag_out = self.weight_i * imag_norm + self.bias_i
		else:
			real_out = real_norm
			imag_out = imag_norm
		
		return real_out, imag_out


class ComplexPositionalEncoding(nn.Module):
	def __init__(self, d_model_complex: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()
		# 为实部和虚部创建独立的位置编码
		self.pe_real = PositionalEncoding(d_model_complex, dropout, max_len)
		self.pe_imag = PositionalEncoding(d_model_complex, dropout, max_len)
		# 为虚部引入相位偏移，使其与实部有所区别
		self.phase_shift = nn.Parameter(torch.randn(d_model_complex) * 0.1)
		
	def forward(self, real, imag, pos: int = 0):
		# 对实部使用标准位置编码
		real_encoded = self.pe_real(real, pos)
		# 对虚部使用带相位偏移的位置编码
		imag_with_shift = imag + self.phase_shift.unsqueeze(0).unsqueeze(0)
		imag_encoded = self.pe_imag(imag_with_shift, pos)
		return real_encoded, imag_encoded


class ComplexMultiheadAttention(nn.Module):
	def __init__(self, d_model_complex: int, nhead: int, dropout: float = 0.0):
		super().__init__()
		assert d_model_complex % nhead == 0
		self.d_model = d_model_complex
		self.nhead = nhead
		self.head_dim = d_model_complex // nhead
		self.q_proj = ComplexLinear(d_model_complex, d_model_complex)
		self.k_proj = ComplexLinear(d_model_complex, d_model_complex)
		self.v_proj = ComplexLinear(d_model_complex, d_model_complex)
		self.out_proj = ComplexLinear(d_model_complex, d_model_complex)
		self.dropout = nn.Dropout(dropout)
		# 移除这里的LayerNorm，应该在上层处理残差连接和归一化
	def _reshape_heads(self, x):
		# x: [T, B, C] -> [B, H, T, D] where D = C/H
		T, B, C = x.shape
		D = C // self.nhead
		# 重新塑形为多头格式
		x = x.permute(1, 0, 2)  # [B, T, C]
		x = x.contiguous().view(B, T, self.nhead, D)  # [B, T, H, D]  
		x = x.permute(0, 2, 1, 3)  # [B, H, T, D]
		return x
	def _merge_heads(self, x):
		# x: [B, H, T, D] -> [T, B, C] where C = H*D
		B, H, T, D = x.shape
		x = x.permute(0, 2, 1, 3)  # [B, T, H, D]
		x = x.contiguous().view(B, T, H * D)  # [B, T, C]
		x = x.permute(1, 0, 2)  # [T, B, C]
		return x
	def forward(self, q_r, q_i, k_r, k_i, v_r, v_i):
		# project
		q_r, q_i = self.q_proj(q_r, q_i)
		k_r, k_i = self.k_proj(k_r, k_i)
		v_r, v_i = self.v_proj(v_r, v_i)
		# [B,H,T,D]
		q_rh = self._reshape_heads(q_r); q_ih = self._reshape_heads(q_i)
		k_rh = self._reshape_heads(k_r); k_ih = self._reshape_heads(k_i)
		v_rh = self._reshape_heads(v_r); v_ih = self._reshape_heads(v_i)
		# 改进复数注意力计算：使用复数相似度
		# scores = Re(q * k^H) = Re(q_r * k_r^T + q_i * k_i^T) + Im(q_i * k_r^T - q_r * k_i^T)
		# 这里只取实部作为注意力分数
		scores = torch.matmul(q_rh, k_rh.transpose(-1, -2)) + torch.matmul(q_ih, k_ih.transpose(-1, -2))
		scores = scores / math.sqrt(self.head_dim)
		attn = torch.softmax(scores, dim=-1)
		attn = self.dropout(attn)
		# weighted sum for values
		out_rh = torch.matmul(attn, v_rh)
		out_ih = torch.matmul(attn, v_ih)
		# merge heads
		out_r = self._merge_heads(out_rh)
		out_i = self._merge_heads(out_ih)
		# final proj
		out_r, out_i = self.out_proj(out_r, out_i)
		return out_r, out_i


class ComplexTransformerEncoderLayer(nn.Module):
	def __init__(self, d_model_complex: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0, activation_type: str = 'learnable_cosine'):
		super().__init__()
		self.self_attn = ComplexMultiheadAttention(d_model_complex, nhead, dropout)
		self.lin1 = ComplexLinear(d_model_complex, dim_feedforward)
		self.lin2 = ComplexLinear(dim_feedforward, d_model_complex)
		self.drop_attn = ComplexDropout(dropout)
		self.drop_ff = ComplexDropout(dropout)
		# 只需要一个激活函数，在FF网络中间 - 使用可配置的激活函数
		self.act = get_activation_function(activation_type)
		# Pre-LN架构的LayerNorm层
		self.norm1 = ComplexLayerNorm(d_model_complex)
		self.norm2 = ComplexLayerNorm(d_model_complex)
	def forward(self, src_r, src_i):
		# Pre-LN Self-attention: LayerNorm → Attention → Residual
		norm_r, norm_i = self.norm1(src_r, src_i)
		attn_r, attn_i = self.self_attn(norm_r, norm_i, norm_r, norm_i, norm_r, norm_i)
		attn_r, attn_i = self.drop_attn(attn_r, attn_i)
		src_r = src_r + attn_r
		src_i = src_i + attn_i
		
		# Pre-LN Feed-forward: LayerNorm → Linear → Activation → Dropout → Linear → Residual
		norm_r, norm_i = self.norm2(src_r, src_i)
		ff_r, ff_i = self.lin1(norm_r, norm_i)
		ff_r, ff_i = self.act(ff_r, ff_i)  # 激活函数在中间
		ff_r, ff_i = self.drop_ff(ff_r, ff_i)
		ff_r, ff_i = self.lin2(ff_r, ff_i)  # 第二个Linear没有激活函数
		src_r = src_r + ff_r
		src_i = src_i + ff_i
		return src_r, src_i


class ComplexTransformerDecoderLayer(nn.Module):
	def __init__(self, d_model_complex: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0, activation_type: str = 'learnable_cosine'):
		super().__init__()
		self.self_attn = ComplexMultiheadAttention(d_model_complex, nhead, dropout)
		self.cross_attn = ComplexMultiheadAttention(d_model_complex, nhead, dropout)
		self.lin1 = ComplexLinear(d_model_complex, dim_feedforward)
		self.lin2 = ComplexLinear(dim_feedforward, d_model_complex)
		self.drop_attn = ComplexDropout(dropout)
		self.drop_cross = ComplexDropout(dropout)
		self.drop_ff = ComplexDropout(dropout)
		# 只需要一个激活函数，在FF网络中间 - 使用可配置的激活函数
		self.act = get_activation_function(activation_type)
		# Pre-LN架构的LayerNorm层
		self.norm1 = ComplexLayerNorm(d_model_complex)
		self.norm2 = ComplexLayerNorm(d_model_complex)
		self.norm3 = ComplexLayerNorm(d_model_complex)
	def forward(self, tgt_r, tgt_i, mem_r, mem_i):
		# Pre-LN Self-attention: LayerNorm → Self-Attention → Residual
		norm_r, norm_i = self.norm1(tgt_r, tgt_i)
		sr_r, sr_i = self.self_attn(norm_r, norm_i, norm_r, norm_i, norm_r, norm_i)
		sr_r, sr_i = self.drop_attn(sr_r, sr_i)
		tgt_r = tgt_r + sr_r
		tgt_i = tgt_i + sr_i
		
		# Pre-LN Cross-attention: LayerNorm → Cross-Attention → Residual
		norm_r, norm_i = self.norm2(tgt_r, tgt_i)
		cr_r, cr_i = self.cross_attn(norm_r, norm_i, mem_r, mem_i, mem_r, mem_i)
		cr_r, cr_i = self.drop_cross(cr_r, cr_i)
		tgt_r = tgt_r + cr_r
		tgt_i = tgt_i + cr_i
		
		# Pre-LN Feed-forward: LayerNorm → Linear → Activation → Dropout → Linear → Residual
		norm_r, norm_i = self.norm3(tgt_r, tgt_i)
		ff_r, ff_i = self.lin1(norm_r, norm_i)
		ff_r, ff_i = self.act(ff_r, ff_i)  # 激活函数在中间
		ff_r, ff_i = self.drop_ff(ff_r, ff_i)
		ff_r, ff_i = self.lin2(ff_r, ff_i)  # 第二个Linear没有激活函数
		tgt_r = tgt_r + ff_r
		tgt_i = tgt_i + ff_i
		return tgt_r, tgt_i


class TranAD_QC(nn.Module):
	"""Complex-valued TranAD variant treating adjacent features as (real, imag) pairs.
	Maintains the same external interface: forward(src[w,B,feats], tgt[1,B,feats]) -> outputs [1,B,feats].
	"""
	def __init__(self, feats: int, activation_type: str = 'learnable_cosine', loss_type: str = 'complex_aware', focus_type: str = 'complex_aware'):
		super().__init__()
		assert feats % 2 == 0, 'TranAD_QC expects even number of features (adjacent pairs as complex).'
		# Create a descriptive name that includes ablation study parameters
		ablation_suffix = f"_{activation_type}_{loss_type}_{focus_type}"
		self.name = f'TranAD_QC{ablation_suffix}'
		self.lr = lr
		self.batch = 512
		self.n_feats = feats
		self.n_feats_c = feats // 2
		self.n_window = 10
		
		# Store ablation study parameters
		self.activation_type = activation_type
		self.loss_type = loss_type
		self.focus_type = focus_type
		
		# After concatenating src and c in complex space -> 大幅增加模型维度以提供更强的表达能力
		self.d_model_c = max(128, 8 * self.n_feats_c)  # 进一步增大d_model_c，至少128维
		# Complex positional encoding and transformer layers
		self.pos_enc_c = ComplexPositionalEncoding(self.d_model_c, 0.1, self.n_window)
		# 重新平衡head配置：确保合理的head数量，每个head也有足够的维度
		min_head_dim = 16  # 降低最小head维度要求
		max_heads = self.d_model_c // min_head_dim
		# 更智能的head数量计算：基于特征数量，但确保不会太少
		suggested_heads = max(2, min(8, self.n_feats_c))  # 2-8个head之间
		optimal_nhead = min(max_heads, suggested_heads)
		# 确保nhead能被d_model_c整除
		while self.d_model_c % optimal_nhead != 0:
			optimal_nhead -= 1
		actual_head_dim = self.d_model_c // optimal_nhead
		print(f"[TranAD_QC] d_model_c={self.d_model_c}, nhead={optimal_nhead}, head_dim={actual_head_dim}")
		
		# 大幅增加feedforward维度以匹配更大的模型
		ff_dim = max(256, self.d_model_c * 4)
		
		# Create layers with configurable activation function
		self.enc_layer_c = ComplexTransformerEncoderLayer(d_model_complex=self.d_model_c, nhead=optimal_nhead, dim_feedforward=ff_dim, dropout=0.1, activation_type=activation_type)
		self.encoder_c = nn.ModuleList([self.enc_layer_c])
		self.dec1_layer_c = ComplexTransformerDecoderLayer(d_model_complex=self.d_model_c, nhead=optimal_nhead, dim_feedforward=ff_dim, dropout=0.1, activation_type=activation_type)
		self.dec2_layer_c = ComplexTransformerDecoderLayer(d_model_complex=self.d_model_c, nhead=optimal_nhead, dim_feedforward=ff_dim, dropout=0.1, activation_type=activation_type)
		# Output complex linear heads mapping back to feats_c, then sigmoid
		self.fcn1_c = ComplexLinear(self.d_model_c, self.n_feats_c)
		self.fcn2_c = ComplexLinear(self.d_model_c, self.n_feats_c)
		
		# 添加输入和目标的投影层，将原始特征投影到更大的模型维度
		input_dim_c = 2 * self.n_feats_c  # concatenated src and c的复数维度
		tgt_dim_c = self.n_feats_c  # target的复数维度
		self.input_proj = ComplexLinear(input_dim_c, self.d_model_c)
		self.tgt_proj = ComplexLinear(tgt_dim_c, self.d_model_c)
		
		# Pre-LN架构需要输入层归一化
		self.input_norm = ComplexLayerNorm(self.d_model_c)
		
		# Use configurable activation functions for output layers
		self.out_act1 = get_activation_function(activation_type)
		self.out_act2 = get_activation_function(activation_type)
		self.sigmoid = ComplexSigmoid()
		# Add LayerNorm for output stabilization
		self.out_norm1 = ComplexLayerNorm(self.n_feats_c)
		self.out_norm2 = ComplexLayerNorm(self.n_feats_c)
		# Optional classifier - 支持不同版本
		try:
			from src.constants import args
			classifier_version = getattr(args, 'clf_version', 'v1')
		except:
			classifier_version = 'v1'  # 默认版本
			
		if classifier_version == 'v2':
			self.classifier = ComplexClfHeadV2(self.n_feats, dropout_p=0.3)
		elif classifier_version == 'v3':
			self.classifier = ComplexClfHeadV3(self.n_feats, dropout_p=0.3)
		else:
			self.classifier = ComplexClfHead(self.n_feats)  # 默认版本
		print(f"[TranAD_QC] 使用分类头版本: {classifier_version}")
		# MAML related parameters (keep parity with TranAD_C)
		self.meta_lr = 0.001
		# Blend weight for focus: angle vs amplitude
		self.focus_alpha = 0.5
	def _encode(self, src, c, tgt):
		# Split to complex, concatenate src and c along complex channels
		sr, si = _split_complex_adjacent(src)
		cr, ci = _split_complex_adjacent(c)
		x_r = torch.cat([sr, cr], dim=2)
		x_i = torch.cat([si, ci], dim=2)
		
		# 使用预定义的投影层扩展到模型维度
		x_r, x_i = self.input_proj(x_r, x_i)
		
		# scale and add positional encoding
		scale = math.sqrt(self.d_model_c)
		x_r = x_r * scale; x_i = x_i * scale
		x_r, x_i = self.pos_enc_c(x_r, x_i)
		
		# Pre-LN架构：在进入transformer前归一化
		x_r, x_i = self.input_norm(x_r, x_i)
		
		# single-layer encoder
		mem_r, mem_i = x_r, x_i
		for layer in self.encoder_c:
			mem_r, mem_i = layer(mem_r, mem_i)
		# prepare tgt in complex space and project to model dimension
		tr, ti = _split_complex_adjacent(tgt)
		tr, ti = self.tgt_proj(tr, ti)
		return tr, ti, mem_r, mem_i
	def forward_phase1(self, src, tgt):
		c = torch.zeros_like(src)
		tr, ti, mem_r, mem_i = self._encode(src, c, tgt)
		# two decoders for adversarial-like dual outputs
		o1_r, o1_i = self.dec1_layer_c(tr, ti, mem_r, mem_i)
		o2_r, o2_i = self.dec2_layer_c(tr, ti, mem_r, mem_i)
		# heads
		o1_r, o1_i = self.fcn1_c(o1_r, o1_i)
		o2_r, o2_i = self.fcn2_c(o2_r, o2_i)
		# activation per part - ComplexCosineActivation handles both real and imag together
		o1_r, o1_i = self.out_act1(o1_r, o1_i)
		o2_r, o2_i = self.out_act2(o2_r, o2_i)
		# o1_r, o1_i = self.sigmoid(o1_r, o1_i)
		# o2_r, o2_i = self.sigmoid(o2_r, o2_i)
		# LayerNorm for output stabilization
		# o1_r, o1_i = self.out_norm1(o1_r, o1_i)
		# o2_r, o2_i = self.out_norm2(o2_r, o2_i)
		# merge back to real feature layout
		o1 = _merge_complex_adjacent(o1_r, o1_i)
		o2 = _merge_complex_adjacent(o2_r, o2_i)
		return o1, o2
	def forward_phase2(self, src, tgt, focus_score):
		tr, ti, mem_r, mem_i = self._encode(src, focus_score, tgt)
		o2h_r, o2h_i = self.dec2_layer_c(tr, ti, mem_r, mem_i)
		o2h_r, o2h_i = self.fcn2_c(o2h_r, o2h_i)
		o2h_r, o2h_i = self.out_act2(o2h_r, o2h_i)
		# o2h_r, o2h_i = self.sigmoid(o2h_r, o2h_i)
		# o2h_r, o2h_i = self.out_norm2(o2h_r, o2h_i)
		o2_hat = _merge_complex_adjacent(o2h_r, o2h_i)
		return o2_hat
	def forward(self, src, tgt, return_all: bool = False):
		o1, o2 = self.forward_phase1(src, tgt)
		
		if self.focus_type == 'complex_aware':
			# Original complex-aware focus computation
			sr, si = _split_complex_adjacent(src)
			o1r, o1i = _split_complex_adjacent(o1)
			
			# 添加小常数避免除零
			eps = 1e-12
			
			# Angle dissimilarity (cosine-like on complex vectors; global over channels)
			num = (sr * o1r + si * o1i)
			norm_src_vec = torch.sqrt((sr * sr + si * si) + eps)
			norm_o1_vec = torch.sqrt((o1r * o1r + o1i * o1i) + eps)
			rho = num / (norm_src_vec * norm_o1_vec + eps)
			rho = rho.mean(dim=2, keepdim=True)
			rho = torch.clamp(rho, -1.0, 1.0)
			angle_dist = 0.5 * (1.0 - rho)  # [w,B,1]
			
			# Amplitude contrast (per-channel), then average to scalar
			mag_src_ch = torch.sqrt(sr * sr + si * si + eps)
			mag_o1_ch = torch.sqrt(o1r * o1r + o1i * o1i + eps)
			kappa_ch = torch.abs(mag_o1_ch - mag_src_ch) / (mag_o1_ch + mag_src_ch + eps)  # [w,B,C]
			amp_contrast = kappa_ch.mean(dim=2, keepdim=True)  # [w,B,1]
			
			# Blend
			focus_scalar = self.focus_alpha * angle_dist + (1.0 - self.focus_alpha) * amp_contrast
			focus = focus_scalar.repeat(1, 1, self.n_feats_c)
			zeros = torch.zeros_like(focus)
			c_complex = _merge_complex_adjacent(focus, zeros)
		elif self.focus_type == 'mse':
			# Simple MSE-based focus score
			c_complex = (o1 - src) ** 2
		else:
			raise ValueError(f"Unknown focus type: {self.focus_type}")
			
		o2_hat = self.forward_phase2(src, tgt, c_complex)
		if return_all:
			return o1, o2, o2_hat
		else:
			return o1, o2_hat
	def extract_error_features(self, src, tgt, mode='simple'):
		"""Return last-step features for classification [B, feats]:
		mode='simple': 直接返回o2_hat (已包含focus信息)
		mode='complex': 返回复数感知的误差特征 (kappa+psi交替)
		"""
		with torch.no_grad():
			out = self.forward(src, tgt, return_all=True)
			if isinstance(out, tuple) and len(out) == 3:
				_, _, o2_hat = out
			else:
				_, o2_hat = out
			
			if mode == 'simple':
				# 直接返回o2_hat，它已经包含了对抗训练的focus信息
				return o2_hat[0]  # [B, feats]
			elif mode == 'complex':
				# 返回复数感知的误差特征
				sr, si = _split_complex_adjacent(tgt)      # [1,B,C]
				o2r, o2i = _split_complex_adjacent(o2_hat) # [1,B,C]
				
				# 计算幅度
				mag_t = torch.sqrt(sr * sr + si * si + 1e-12)
				mag_o = torch.sqrt(o2r * o2r + o2i * o2i + 1e-12)
				
				# 幅度对比度 kappa ∈ [0,1]
				kappa = torch.abs(mag_o - mag_t) / (mag_o + mag_t + 1e-12)  # [1,B,C]
				
				# 角度相似性
				cos_delta = (o2r * sr + o2i * si) / (mag_o * mag_t + 1e-12)
				cos_delta = torch.clamp(cos_delta, -1.0, 1.0)
				psi = 0.5 * (1.0 - cos_delta)  # [1,B,C] 角度差异 ∈ [0,1]
				
				# 将kappa和psi交替排列，形成最终特征
				feat = _merge_complex_adjacent(kappa, psi)[0]  # [B, feats]
				return feat
			else:
				raise ValueError(f"Unsupported mode: {mode}")
	
	def predict_scores(self, src, tgt, mode='simple'):
		feat = self.extract_error_features(src, tgt, mode=mode)
		logits = self.classifier(feat)
		return logits
	def get_encoder_params(self):
		"""Parameters of complex positional encoding and encoder layers."""
		params = []
		params.extend(self.pos_enc_c.parameters())
		params.extend(self.input_proj.parameters())  # 包含输入投影层
		params.extend(self.tgt_proj.parameters())    # 包含目标投影层
		for layer in self.encoder_c:
			params.extend(layer.parameters())
		return list(params)
	def get_decoder1_params(self):
		"""Parameters of first decoder branch (decoder1 + head)."""
		params = []
		params.extend(self.dec1_layer_c.parameters())
		params.extend(self.fcn1_c.parameters())
		return list(params)
	def get_decoder2_params(self):
		"""Parameters of second decoder branch (decoder2 + head)."""
		params = []
		params.extend(self.dec2_layer_c.parameters())
		params.extend(self.fcn2_c.parameters())
		return list(params)
	def clone(self):
		"""Create a copy for MAML-like procedures."""
		clone = TranAD_QC(self.n_feats, self.activation_type, self.loss_type, self.focus_type)
		clone.load_state_dict(self.state_dict())
		return clone

class ComplexClfHeadV3(nn.Module):
	"""复数注意力分类头：自适应特征重要性 + 复数池化"""
	def __init__(self, feats: int, hidden_complex: int = None, dropout_p: float = 0.2):
		super().__init__()
		assert feats % 2 == 0, 'ComplexClfHead expects even feature size (adjacent real/imag).'
		self.n_complex = feats // 2
		self.hidden_complex = hidden_complex if hidden_complex is not None else max(32, self.n_complex * 2)
		
		# 特征提取
		self.feature_extractor = nn.Sequential(
			ComplexLinear(self.n_complex, self.hidden_complex),
			ComplexCosineActivation(omega_init=1.0, phi_init=0.0),
			ComplexLayerNorm(self.hidden_complex)
		)
		
		# 复数自注意力机制（简化版，针对特征维度）
		self.attention_q = ComplexLinear(self.hidden_complex, self.hidden_complex // 4)
		self.attention_k = ComplexLinear(self.hidden_complex, self.hidden_complex // 4)
		self.attention_v = ComplexLinear(self.hidden_complex, self.hidden_complex)
		
		# 多种复数池化策略
		self.mag_pool = nn.AdaptiveAvgPool1d(1)
		self.phase_pool = nn.AdaptiveAvgPool1d(1)
		
		# 最终分类器
		self.classifier = nn.Sequential(
			nn.Linear(self.hidden_complex * 3, self.hidden_complex),  # 3倍是因为原始+幅度+相位
			nn.ReLU(),
			nn.Dropout(dropout_p),
			nn.Linear(self.hidden_complex, self.hidden_complex // 2),
			nn.ReLU(), 
			nn.Dropout(dropout_p),
			nn.Linear(self.hidden_complex // 2, 1)
		)
		
	def complex_attention(self, h_r, h_i):
		"""复数自注意力"""
		q_r, q_i = self.attention_q(h_r, h_i)
		k_r, k_i = self.attention_k(h_r, h_i)
		v_r, v_i = self.attention_v(h_r, h_i)
		
		# 复数attention score: Re(q * k^H)
		scores = q_r * k_r + q_i * k_i  # [B, hidden//4]
		attention_weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, hidden//4, 1]
		
		# 应用注意力权重
		attended_r = v_r * attention_weights
		attended_i = v_i * attention_weights
		
		return attended_r, attended_i
		
	def forward(self, x):
		real, imag = _split_complex_adjacent(x)
		
		# 特征提取
		h_r, h_i = self.feature_extractor[0](real, imag)
		h_r, h_i = self.feature_extractor[1](h_r, h_i)
		h_r, h_i = self.feature_extractor[2](h_r, h_i)
		
		# 自注意力
		att_r, att_i = self.complex_attention(h_r, h_i)
		
		# 多种特征表示
		# 1. 原始复数特征的全局平均
		orig_feat = torch.cat([h_r.mean(dim=-1), h_i.mean(dim=-1)], dim=-1)
		
		# 2. 幅度特征
		magnitude = torch.sqrt(att_r * att_r + att_i * att_i + 1e-12)
		mag_feat = magnitude.mean(dim=-1)
		
		# 3. 相位特征
		phase = torch.atan2(att_i + 1e-12, att_r + 1e-12)
		phase_feat = phase.mean(dim=-1)
		
		# 融合所有特征
		combined_feat = torch.cat([orig_feat, mag_feat, phase_feat], dim=-1)
		
		# 分类
		logits = self.classifier(combined_feat)
		return logits.squeeze(-1)

class ComplexReconstructionLoss(nn.Module):
	"""Complex-aware reconstruction loss producing per-feature values [T,B,feats].
	For each complex channel c, compute blended angle and amplitude dissimilarity:
	  loss_c = alpha * 0.5*(1 - cosΔ) + (1-alpha) * | |p|-|t| | / (|p|+|t|+eps)
	Then interleave to real layout by duplicating onto (r_c, i_c).
	"""
	def __init__(self, alpha: float = 0.9):
		super().__init__()
		self.alpha = alpha
	def forward(self, pred, target):
		pr, pi = _split_complex_adjacent(pred)
		tr, ti = _split_complex_adjacent(target)
		
		# 添加小常数避免除零
		eps = 1e-12
		
		mag_p = torch.sqrt(pr * pr + pi * pi + eps)
		mag_t = torch.sqrt(tr * tr + ti * ti + eps)
		
		# angle part
		cos_delta = (pr * tr + pi * ti) / (mag_p * mag_t + eps)
		cos_delta = torch.clamp(cos_delta, -1.0, 1.0)
		psi = 0.5 * (1.0 - cos_delta)
		
		# amplitude part
		kappa = torch.abs(mag_p - mag_t) / (mag_p + mag_t + eps)
		loss_c = self.alpha * psi + (1.0 - self.alpha) * kappa  # [T,B,C]
		loss_full = _merge_complex_adjacent(loss_c, loss_c)     # [T,B,feats]
		return loss_full


class ComplexMSELoss(nn.Module):
	"""Standard complex MSE loss: ||pred - target||² = |pred_r - target_r|² + |pred_i - target_i|²"""
	def __init__(self):
		super().__init__()
	
	def forward(self, pred, target):
		# 正确的复数MSE计算
		# pred和target都是复数张量，形状为 [batch, seq_len, features]
		# 使用_merge_complex_adjacent格式：(r0,i0,r1,i1,...)
		
		# 计算复数差
		diff = pred - target
		
		# 根据_merge_complex_adjacent的格式，实部在偶数索引，虚部在奇数索引
		if diff.shape[-1] % 2 == 0:  # 确保是复数特征
			real_diff = diff[..., ::2]  # 实部 (r0, r1, r2, ...)
			imag_diff = diff[..., 1::2]  # 虚部 (i0, i1, i2, ...)
			
			# 复数MSE = |diff|² = real_diff² + imag_diff²
			# 对于复数 z = a + j*b，|z|² = a² + b²
			mse_loss = real_diff**2 + imag_diff**2
		else:
			# 如果不是复数特征，直接计算MSE
			mse_loss = diff**2
		
		return mse_loss


def get_activation_function(activation_type: str):
	"""Factory function to create activation functions based on type."""
	if activation_type == 'learnable_cosine':
		return ComplexCosineActivation(omega_init=1.0, phi_init=0.0)
	elif activation_type == 'relu':
		return ComplexReLU()
	elif activation_type == 'leaky_relu':
		return ComplexLeakyReLU()
	elif activation_type == 'fixed_cosine':
		return ComplexFixedCosineActivation(omega=1.0, phi=0.0)
	else:
		raise ValueError(f"Unknown activation type: {activation_type}")


def get_loss_function(loss_type: str, alpha: float = 0.9):
	"""Factory function to create loss functions based on type."""
	if loss_type == 'complex_aware':
		return ComplexReconstructionLoss(alpha=alpha)
	elif loss_type == 'mse':
		return ComplexMSELoss()
	else:
		raise ValueError(f"Unknown loss type: {loss_type}")
