import os
import glob
import numpy as np
import pandas as pd


def read_raw_csv(file_path: str) -> pd.DataFrame:
    """读取原始 CSV（跳过前 4 行），按原脚本方式丢弃第 0、28 列，丢弃最后一行。

    返回的 DataFrame 不带列名，索引连续。
    列意义（按你原脚本约定）：
      - 列 1：字符串/标记列（包含 'WER' 表示异常）
      - 列 2 之后：数值特征
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    lines = content.split('\n')
    # 跳过前 4 行，按照逗号切分
    rows = [line.split(',') for line in lines[4:] if line.strip() != '']
    df = pd.DataFrame(rows)
    # 丢弃第 0、28 列（若存在）
    drop_cols = []
    for col in (0, 28):
        if col in df.columns:
            drop_cols.append(col)
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    # 丢弃最后一行（通常是汇总或无效行）
    if len(df) > 0:
        df = df.iloc[:-1, :]
    df.reset_index(drop=True, inplace=True)
    return df


def standardize_per_file(features: np.ndarray) -> np.ndarray:
    """模仿原脚本 scaler_df：对单文件内的数值特征做 z-score 标准化。"""
    eps = 1e-6
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < eps, eps, std)
    return (features - mean) / std


def preprocess_my_data(
    input_dir: str = os.path.join('data', 'my_data', 'group21(1)', 'group21'),
    output_dir: str = os.path.join('processed', 'my_data'),
    test_ratio: float = 0.2,
    expand_label: int = 20,
):
    """将自定义数据转换为标准格式（不在此阶段窗口化）：

    - 训练集 train：仅正常行（剔除 invalid 行）。
    - 测试集 test：覆盖所有异常（基于 'WER' 行并做 ±expand_label 扩张），并扩张到约 test_ratio 比例；不跨文件。
    - 标签 labels：对 test 的每行复制行级标签，形成与 test 同形状的 0/1 矩阵。
    - 另存 train_file_ids/test_file_ids，训练阶段用于避免跨文件窗口。
    - 额外保存分类头训练/测试集（clf_train/clf_labels/clf_file_ids 与 clf_test/*），二者严格不重叠；测试集中异常最多 10%。
    """
    file_glob = os.path.join(input_dir, '*.csv')
    files = sorted(glob.glob(file_glob))
    if len(files) == 0:
        raise FileNotFoundError(f'未找到 CSV：{file_glob}')

    feature_blocks = []  # 按文件得到的特征（标准化后）
    label_blocks = []    # 按文件得到的行级标签（0/1），稍后复制到各特征列
    file_id_blocks = []  # 记录每行所属文件 id，便于训练阶段避免跨文件窗口

    for fp in files:
        try:
            df = read_raw_csv(fp)
            if df.shape[1] < 3:
                # 至少需要 [?, 标记列, 至少1个特征]
                continue
            # 第 1 列：用于字符串标记（WER / invaild / invalid）
            mark_col = df.columns[0]
            mark_series = df[mark_col].astype(str)
            # 过滤掉 invaild/invalid 行
            invalid_mask = mark_series.str.contains('invaild', case=False, na=False) | \
                           mark_series.str.contains('invalid', case=False, na=False)
            df = df.loc[~invalid_mask].reset_index(drop=True)
            mark_series = df[mark_col].astype(str)

            # 数值特征：第 2 列及之后（转数值，非数字->NaN，±inf->NaN，最后填0）
            feat_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
            feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            feats = feat_df.values.astype(np.float64)
            # 原 z-score 标准化移除，保留原始数值，后续统一在 train/test 划分后按训练集做 Min-Max
            # feats = standardize_per_file(feats)
            # 再次兜底，消除可能出现的非有限值
            feats[~np.isfinite(feats)] = 0.0

            # 行级粗标签（不在预处理中窗口化）：是否包含 WER
            num_rows = len(df)
            is_wer = mark_series.str.contains('WER', case=False, na=False).values.astype(bool)
            row_labels = is_wer.astype(np.float64)

            feature_blocks.append(feats)
            label_blocks.append(row_labels)
            file_id_blocks.append(np.full(len(row_labels), len(file_id_blocks), dtype=np.int64))
        except Exception:
            # 忽略异常文件，继续处理
            continue

    if len(feature_blocks) == 0:
        raise RuntimeError('未成功解析任何 CSV 文件为特征块。')

    features_full = np.vstack(feature_blocks)
    # 兜底清洗：将全量特征中的非有限值置零
    features_full[~np.isfinite(features_full)] = 0.0
    row_labels_full = np.concatenate(label_blocks)
    file_ids_full = np.concatenate(file_id_blocks)

    # 构建文件范围，做异常膨胀（±expand_label），并抽样约 test_ratio 的测试集
    T, D = features_full.shape
    unique_fids = np.unique(file_ids_full)
    file_ranges = {}
    for fid in unique_fids:
        idxs = np.where(file_ids_full == fid)[0]
        if idxs.size:
            file_ranges[fid] = (idxs[0], idxs[-1])
    # 异常膨胀到 expanded_labels
    expanded_labels = np.zeros(T, dtype=bool)
    pos = np.where(row_labels_full > 0)[0]
    for idx in pos:
        fid = file_ids_full[idx]
        Lf, Rf = file_ranges[fid]
        L = max(Lf, idx - expand_label)
        R = min(Rf + 1, idx + expand_label + 1)
        expanded_labels[L:R] = True
    # 初始测试掩码覆盖所有 expanded_labels，然后按文件内部扩张，直到达到目标比例
    target = max(int(np.ceil(test_ratio * T)), int(expanded_labels.sum()))
    test_mask = expanded_labels.copy()
    def grow_in_file(fid: int, step: int = 100) -> int:
        nonlocal test_mask
        Lf, Rf = file_ranges[fid]
        covered = np.where(test_mask[Lf:Rf+1])[0]
        if covered.size == 0:
            center = (Lf + Rf) // 2
            L = max(Lf, center - step)
            R = min(Rf + 1, center + step)
            new = (~test_mask[L:R]).sum()
            test_mask[L:R] = True
            return int(new)
        Lc = Lf + covered.min()
        Rc = Lf + covered.max()
        L = max(Lf, Lc - step)
        R = min(Rf + 1, Rc + step)
        new = (~test_mask[L:R]).sum()
        test_mask[L:R] = True
        return int(new)
    need = target - int(test_mask.sum())
    while need > 0:
        grew = 0
        for fid in unique_fids:
            grew += grow_in_file(fid, step=100)
            need = target - int(test_mask.sum())
            if need <= 0:
                break
        if grew == 0:
            break
    # 划分集合（注意此时还未做缩放）
    train_mask = (~test_mask) & (row_labels_full == 0)

    # 基于训练集统计做 Min-Max 归一化，并应用到 train/test
    eps = 1e-6
    train_raw = features_full[train_mask]
    test_raw = features_full[test_mask]
    fmin = np.min(train_raw, axis=0)
    fmax = np.max(train_raw, axis=0)
    denom = fmax - fmin
    denom = np.where(denom < eps, eps, denom)

    train = (train_raw - fmin) / denom
    test = (test_raw - fmin) / denom
    # 裁剪到 [0, 1]
    train = np.clip(train, 0.0, 1.0)
    test = np.clip(test, 0.0, 1.0)

    # 构造 test 的逐特征标签矩阵
    labels = np.zeros((test.shape[0], D), dtype=np.float64)
    test_indices = np.where(test_mask)[0]
    pos_map = {g_idx: local for local, g_idx in enumerate(test_indices)}
    for g_idx in np.where(expanded_labels)[0]:
        if g_idx in pos_map:
            labels[pos_map[g_idx], :] = 1.0

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'train.npy'), train)
    # 原 test/labels 保存移动到分割完成后（见下方）
    # 保存文件 id，训练阶段用于避免跨文件取窗口，以及测试索引
    np.save(os.path.join(output_dir, 'train_file_ids.npy'), file_ids_full[train_mask])
    # np.save(os.path.join(output_dir, 'test_file_ids.npy'), file_ids_full[test_mask])
    np.save(os.path.join(output_dir, 'test_global_indices.npy'), test_indices)
    # 保存归一化参数（便于复现和线上一致性）
    np.save(os.path.join(output_dir, 'scaler_min.npy'), fmin)
    np.save(os.path.join(output_dir, 'scaler_max.npy'), fmax)

    # 额外划分分类头训练/测试集（不重叠；测试集中异常最多 10%，正常与异常数量尽量平衡）
    row_anom = (labels.sum(axis=1) > 0)
    idx_anom = np.where(row_anom)[0]
    idx_norm = np.where(~row_anom)[0]
    rng = np.random.default_rng(42)
    k_anom_test = max(1, int(np.floor(0.10 * len(idx_anom)))) if len(idx_anom) > 0 else 0
    anom_test = rng.choice(idx_anom, size=k_anom_test, replace=False) if k_anom_test > 0 else np.array([], dtype=int)
    # 为了平衡，从正常中抽取与异常测试集同等数量（若不足则全取）
    k_norm_test = min(len(idx_norm), len(anom_test))
    norm_test = rng.choice(idx_norm, size=k_norm_test, replace=False) if k_norm_test > 0 else np.array([], dtype=int)
    clf_test_mask = np.zeros(test.shape[0], dtype=bool)
    clf_test_mask[anom_test] = True
    clf_test_mask[norm_test] = True
    clf_train_mask = ~clf_test_mask

    clf_train = test[clf_train_mask]
    clf_labels = labels[clf_train_mask]
    clf_file_ids = file_ids_full[test_mask][clf_train_mask]
    # 最终整体评测用的 test 集（不与分类头训练重叠，异常<=10%）
    final_test = test[clf_test_mask]
    final_labels = labels[clf_test_mask]
    final_test_file_ids = file_ids_full[test_mask][clf_test_mask]

    # 保存分类头训练集
    np.save(os.path.join(output_dir, 'clf_train.npy'), clf_train)
    np.save(os.path.join(output_dir, 'clf_labels.npy'), clf_labels)
    np.save(os.path.join(output_dir, 'clf_file_ids.npy'), clf_file_ids)
    # 保存整体模型测试集到标准命名（供主流程使用）
    np.save(os.path.join(output_dir, 'test.npy'), final_test)
    np.save(os.path.join(output_dir, 'labels.npy'), final_labels)
    np.save(os.path.join(output_dir, 'test_file_ids.npy'), final_test_file_ids)

    return {
        'train_shape': train.shape,
        'test_shape': final_test.shape,
        'labels_shape': final_labels.shape,
        'num_files': len(files),
        'output_dir': output_dir,
        'has_file_ids': True,
        'train_size': int(train.shape[0]),
        'test_size': int(final_test.shape[0]),
        'test_ratio_target': float(test_ratio),
    }


if __name__ == '__main__':
    info = preprocess_my_data()
    print('保存完成:', info)



