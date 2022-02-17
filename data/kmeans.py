import logging
import os
import sys
from tqdm import tqdm
import random 
import joblib

import h5py
import numpy as np
import torch 
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("kmeans")


def get_km_model(n_clusters, init, max_iter, batch_size, tol, max_no_improvement, n_init, reassignment_ratio):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def load_feature_and_apply_km(apply_kmeans, n_clusters, data_dir, data_split, feature):
    input_h5_fpth  = os.path.join(data_dir, f'{data_split}_segment-{feature}_embed-83k-5k-5k.hdf5')
    output_h5_fpth = os.path.join(data_dir, f'{data_split}_segment-{feature}_embed-km{n_clusters}-83k-5k-5k.hdf5')
    word_list_fpth = os.path.join(data_dir, f'{data_split}_segment-{feature}_word_list-83k-5k-5k.npy')
    word_list_keys = list(np.load(word_list_fpth, allow_pickle=True)[0].keys())
    logging.info(f'Loading feature from h5py file {input_h5_fpth}')
    logging.info(f'Writing feature to h5py file {output_h5_fpth}')
    feature_h5_obj = h5py.File(input_h5_fpth, "r")
    feature_km_h5_obj = h5py.File(output_h5_fpth, "w")

    for tmp_idx in tqdm(word_list_keys):
        tmp_feat = feature_h5_obj[str(tmp_idx)][:] 
        lab = apply_kmeans(tmp_feat).tolist()
        feature_km_h5_obj.create_dataset(str(tmp_idx), data=lab)
    feature_h5_obj.close(), feature_km_h5_obj.close()


def load_feature_and_train_km(km_model, data_dir, feature, percentage=0.1):
    h5_fpth = os.path.join(data_dir, f'train_segment-{feature}_embed-83k-5k-5k.hdf5')
    word_list_fpth = os.path.join(data_dir, f'train_segment-{feature}_word_list-83k-5k-5k.npy')
    logging.info(f'Loading feature from h5py file {h5_fpth}')
    h5_obj = h5py.File(h5_fpth, "r")
    word_list_keys = list(np.load(word_list_fpth, allow_pickle=True)[0].keys())
    random.shuffle(word_list_keys)
    max_frame = int(20e6)
    if feature == 'logmelspec':
        total_feat = np.zeros((max_frame, 40), dtype=np.float32)
    else:
        total_feat = np.zeros((max_frame, 768), dtype=np.float32)
    
    cur_frame_num = 0
    for tmp_idx in tqdm(word_list_keys[:int(len(word_list_keys)*percentage)]):
        tmp_feat = h5_obj[str(tmp_idx)][:] 
        tmp_frame_num = tmp_feat.shape[0]
        if cur_frame_num+tmp_frame_num >= max_frame: 
            break 
        total_feat[cur_frame_num:cur_frame_num+tmp_frame_num, :] = tmp_feat
        cur_frame_num += tmp_frame_num

    total_feat = total_feat[:cur_frame_num, :]
    logging.info(f"fit km_model on feature with shape {total_feat.shape}")
    km_model.fit(total_feat)
    h5_obj.close()

    return km_model, total_feat


def learn_kmeans(data_dir, feature, kmeans_dir, n_clusters, seed, percent, init, 
                 max_iter, batch_size, tol, n_init, reassignment_ratio, max_no_improvement):
    random.seed(seed)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    logger.info('Start kmeans training')
    km_model, feat = load_feature_and_train_km(km_model, data_dir, feature, percent)
    km_path = os.path.join(kmeans_dir, f'train_segment-{feature}_embed-83k-5k-5k.km{n_clusters}')
    logger.info('Storing kmeans model at %s', km_path)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


def apply_kmeans(data_dir, feature, kmeans_dir, n_clusters, seed, percent, init, 
                 max_iter, batch_size, tol, n_init, reassignment_ratio, max_no_improvement):
    km_path = os.path.join(kmeans_dir, f'train_segment-{feature}_embed-83k-5k-5k.km{n_clusters}')
    logger.info('Loading kmeans model from %s', km_path)
    apply_kmeans = ApplyKmeans(km_path)
    for data_split in ['test', 'val', 'train']:
        load_feature_and_apply_km(apply_kmeans, n_clusters, data_dir, data_split, feature)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--kmeans_dir", type=str)
    parser.add_argument('--feature', '-f', type=str, default='logmelspec', 
                        help='e.g. hubert6, hubert12, hubert_large24')

    parser.add_argument("--n_clusters", type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--percent", default=0.1, type=float, help="sample a subset; between 0-1")
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    args = parser.parse_args()
    logging.info(str(args))
    
    # Step 1: learn kmeans 
    #learn_kmeans(**vars(args))

    # Step 2: apply kmeans 
    #apply_kmeans(**vars(args))
