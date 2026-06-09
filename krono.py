##for colab !pip install -q scikit-learn tqdm

import os
import random
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

##hyperparams, concept labels

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
OFFICIAL = {
    "seed": 42,
    #testing!!
    "classifier_epochs": 5,
    #classifier_epochs": 250,
    "detector_epochs": 5,
    #"detector_epochs": 250,
    "adapt_epochs": 5,
    #"adapt_epochs": 250,
    "batch_size": 32,
    "learning_rate": 4e-4,

    #dream settings
    "lambda_rec": 0.1,
    "lambda_sep": 0.01,
    "lambda_rel": 0.01,
    "lambda_pre": 0.001,
    "sep_margin": 10.0,

    #autoencoder architecture
    "ae_hidden_dim": 512,
    "ae_encoding_dim": 32,

    # Dynamic detector schedule
    "eta": 0.1,

    "num_families": 8,
    "min_family_samples": 50,
}

CONCEPT_NAMES = [
    "privacy_stealing",       # b0
    "sms_call_abuse",         # b1
    "remote_control",         # b2
    "bank_financial_stealing",# b3
    "ransom",                 # b4
    "accessibility_abuse",    # b5
    "privilege_escalation",   # b6
    "stealthy_download",      # b7
    "aggressive_advertising", # b8
    "premium_service",        # b9
]

KRONO_CONCEPT_COLUMNS = {
    "privacy_stealing": [
        "READ_CONTACTS",
        "READ_SMS",
        "READ_CALL_LOG",
        "READ_PHONE_STATE",
        "READ_PHONE_NUMBERS",
        "ACCESS_FINE_LOCATION",
        "ACCESS_COARSE_LOCATION",
        "RECORD_AUDIO",
        "CAMERA",
    ],
    "sms_call_abuse": [
        "SEND_SMS",
        "RECEIVE_SMS",
        "READ_SMS",
        "RECEIVE_MMS",
        "RECEIVE_WAP_PUSH",
        "CALL_PHONE",
        "PROCESS_OUTGOING_CALLS",
        "READ_CALL_LOG",
        "WRITE_CALL_LOG",
    ],
    "remote_control": [
        "INTERNET",
        "ACCESS_NETWORK_STATE",
        "ACCESS_WIFI_STATE",
        "CHANGE_NETWORK_STATE",
        "CHANGE_WIFI_STATE",
        "RECEIVE_BOOT_COMPLETED",
        "WAKE_LOCK",
    ],
    "bank_financial_stealing": [
        "SMS_FINANCIAL_TRANSACTIONS",
        "USE_BIOMETRIC",
        "USE_FINGERPRINT",
        "READ_SMS",
        "READ_PHONE_STATE",
    ],
    "ransom": [
        "BIND_DEVICE_ADMIN",
        "SYSTEM_ALERT_WINDOW",
        "REQUEST_DELETE_PACKAGES",
        "DISABLE_KEYGUARD",
    ],
    "accessibility_abuse": [
        "BIND_ACCESSIBILITY_SERVICE",
    ],
    "privilege_escalation": [
        "INSTALL_PACKAGES",
        "DELETE_PACKAGES",
        "REQUEST_INSTALL_PACKAGES",
        "WRITE_SECURE_SETTINGS",
        "WRITE_SETTINGS",
        "MOUNT_UNMOUNT_FILESYSTEMS",
        "REBOOT",
    ],
    "stealthy_download": [
        "REQUEST_INSTALL_PACKAGES",
        "INSTALL_PACKAGES",
        "INTERNET",
        "WRITE_EXTERNAL_STORAGE",
        "READ_EXTERNAL_STORAGE",
    ],
    "aggressive_advertising": [
        "INTERNET",
        "ACCESS_NETWORK_STATE",
        "VIBRATE",
        "WAKE_LOCK",
        "SYSTEM_ALERT_WINDOW",
    ],
    "premium_service": [
        "SEND_SMS",
        "CALL_PHONE",
        "READ_PHONE_STATE",
        "PROCESS_OUTGOING_CALLS",
    ],
}


def build_krono_concepts_from_features(df):
#returns: concepts: [N, 10], 0/1  ;  concept_mask: [N, 10], 1 is valid label
    
    concepts = np.zeros((len(df), len(CONCEPT_NAMES)), dtype=np.float32)
    concept_mask = np.ones_like(concepts, dtype=np.float32)
    for concept_id, concept_name in enumerate(CONCEPT_NAMES):
        source_cols = KRONO_CONCEPT_COLUMNS[concept_name]
        available_cols = [c for c in source_cols if c in df.columns]
        if len(available_cols) == 0:
            concept_mask[:, concept_id] = 0.0
            continue
        values = df[available_cols].fillna(0).astype(float).values
        concepts[:, concept_id] = (values > 0).any(axis=1).astype(np.float32)
    return concepts, concept_mask

##read data and do hold out split

DATA_PATH = "data/emu_malware_v1.zip"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Cannot find {DATA_PATH}. "
        "missing KronoDroid emulator malware zip file"
    )

df = pd.read_csv(DATA_PATH, compression="zip")
print("Raw KronoDroid shape:", df.shape)

if "MalFamily" not in df.columns:
    raise KeyError("KronoDroid malware data must contain MalFamily.")

df["MalFamily"] = df["MalFamily"].astype(str).str.strip()
df = df[
    df["MalFamily"].notna()
    & (df["MalFamily"] != "")
    & (df["MalFamily"].str.lower() != "nan")
].copy()

family_counts = df["MalFamily"].value_counts()
valid_families = family_counts[
    family_counts >= OFFICIAL["min_family_samples"]
].head(OFFICIAL["num_families"]).index.tolist()

if len(valid_families) < 3:
    raise ValueError(
        "Not enough malware families with enough samples. "
        "Try lowering OFFICIAL['min_family_samples']."
    )

df = df[df["MalFamily"].isin(valid_families)].copy()
df = df.reset_index(drop=True)

family_names = np.array(valid_families)
family_to_id = {name: i for i, name in enumerate(family_names)}
y = df["MalFamily"].map(family_to_id).astype(np.int64).values

DROP_COLUMNS = [
    "Package",
    "Malware",
    "sha256",
    "FirstModDate",
    "LastModDate",
    "MalFamily",
    "Scanners",
    "Detection_Ratio",
]

feature_cols = [
    c for c in df.columns
    if c not in DROP_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
]

if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found.")

X = df[feature_cols].fillna(0).astype(np.float32).values
max_vals = np.maximum(X.max(axis=0, keepdims=True), 1.0)
X = X / max_vals

concepts, concept_mask = build_krono_concepts_from_features(df)

print("Using KronoDroid families:")
for name in family_names:
    print(f"  {name}: {family_counts[name]} samples")
print("X:", X.shape)
print("y:", y.shape)
print("concepts:", concepts.shape)
print("valid concept labels:", concept_mask.sum(axis=0))
print("families:", family_names)

def make_holdout_split(
    X,
    y,
    concepts,
    concept_mask,
    holdout_family,
    test_size=0.2,
    seed=42,
):
    y = np.asarray(y)
    drift_idx = np.where(y == holdout_family)[0]
    known_idx = np.where(y != holdout_family)[0]
    X_known = X[known_idx]
    y_known = y[known_idx]
    c_known = concepts[known_idx]
    m_known = concept_mask[known_idx]
    X_drift = X[drift_idx]
    c_drift = concepts[drift_idx]
    m_drift = concept_mask[drift_idx]

    known_train_idx, known_test_idx = train_test_split(
        np.arange(len(X_known)),
        test_size=test_size,
        random_state=seed,
        stratify=y_known,
    )

    known_families = sorted(np.unique(y_known))
    old_to_new = {old: new for new, old in enumerate(known_families)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    def remap_known(labels):
        return np.array([old_to_new[int(v)] for v in labels], dtype=np.int64)
        
    X_train = X_known[known_train_idx]
    y_train = remap_known(y_known[known_train_idx])
    c_train = c_known[known_train_idx]
    m_train = m_known[known_train_idx]
    X_known_test = X_known[known_test_idx]
    y_known_test = remap_known(y_known[known_test_idx])
    c_known_test = c_known[known_test_idx]
    m_known_test = m_known[known_test_idx]
    num_known_classes = len(known_families)
    drift_new_label = num_known_classes
    X_test = np.concatenate([X_known_test, X_drift], axis=0)
    y_test_for_adapt = np.concatenate(
        [
            y_known_test,
            np.full(len(X_drift), drift_new_label, dtype=np.int64),
        ],
        axis=0,
    )

    drift_binary = np.concatenate(
        [
            np.zeros(len(X_known_test), dtype=np.int64),
            np.ones(len(X_drift), dtype=np.int64),
        ],
        axis=0,
    )
    c_test = np.concatenate([c_known_test, c_drift], axis=0)
    m_test = np.concatenate([m_known_test, m_drift], axis=0)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "concept_train": c_train,
        "mask_train": m_train,
        "X_test": X_test,
        "y_test_for_adapt": y_test_for_adapt,
        "drift_binary": drift_binary,
        "concept_test": c_test,
        "mask_test": m_test,
        "num_known_classes": num_known_classes,
        "drift_new_label": drift_new_label,
        "known_families": known_families,
        "holdout_family": holdout_family,
        "old_to_new": old_to_new,
        "new_to_old": new_to_old,
    }

##dream dataset

class DreamDataset(Dataset):
    def __init__(self, X, y, concepts, masks):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.concepts = torch.tensor(concepts, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.concepts[idx], self.masks[idx]

##drebin classifier

class DrebinMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 30),
            nn.ReLU(),
        )
        self.out = nn.Linear(30, num_classes)

    def forward(self, x):
        h = self.feature(x)
        return self.out(h)

##dream tabular autoencoder for drebin

class DreamDetector(nn.Module):
    def __init__(self, input_dim, num_concepts=10, hidden_dim=512, encoding_dim=32):
        super().__init__()

        if encoding_dim < num_concepts:
            raise ValueError("encoding_dim must be >= num_concepts")

        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        self.concept_head = nn.Sigmoid()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        concept_probs = self.concept_head(z[:, :self.num_concepts])
        return x_hat, z, concept_probs


def safe_pairwise_distances(z, eps=1e-10):
    diff = z.unsqueeze(1) - z.unsqueeze(0)
    return torch.sqrt(torch.sum(diff * diff, dim=2) + eps)


def cade_contrastive_separation_loss(z, labels, margin=10.0):
#positive_loss: same-family samples are close
#negative_loss: different-family samples are apart

    if z.size(0) <= 1:
        return z.new_tensor(0.0)
    distances = safe_pairwise_distances(z)
    labels = labels.view(-1, 1)
    positive_mask = labels.eq(labels.T)
    negative_mask = ~positive_mask
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    positive_mask = positive_mask & (~eye)
    losses = []
    if positive_mask.any():
        losses.append(distances[positive_mask].mean())
    if negative_mask.any():
        losses.append(torch.clamp(margin - distances[negative_mask], min=0.0).mean())
    if not losses:
        return z.new_tensor(0.0)
    return sum(losses)

##dream loss

def reconstruction_loss(x_hat, x):
    # Official code uses MSE reconstruction loss.
    return F.mse_loss(x_hat, x)

def concept_presence_loss(concept_probs, concept_labels, concept_mask):
#masked binary cross entropy, concept_mask = 1 is valid

    raw = F.binary_cross_entropy(
        concept_probs.clamp(1e-7, 1.0 - 1e-7),
        concept_labels,
        reduction="none",
    )
    masked = raw * concept_mask
    denom = concept_mask.sum().clamp_min(1.0)
    return masked.sum() / denom

def concept_reliability_loss(classifier, x, x_hat):
#L_rel = CE(M(x), M(x_hat))

    with torch.no_grad():
        original_probs = F.softmax(classifier(x), dim=1)
    reconstructed_log_probs = F.log_softmax(classifier(x_hat), dim=1)
    return -(original_probs * reconstructed_log_probs).sum(dim=1).mean()

def concept_reliability_loss_per_sample(classifier, x, x_hat):
    with torch.no_grad():
        original_probs = F.softmax(classifier(x), dim=1)
    reconstructed_log_probs = F.log_softmax(classifier(x_hat), dim=1)
    return -(original_probs * reconstructed_log_probs).sum(dim=1)

def pseudo_cross_entropy_per_sample(logits):
#use model argmax prediction as pseudo label

    pseudo = logits.argmax(dim=1)
    return F.cross_entropy(logits, pseudo, reduction="none")

##training

def train_classifier(
    model,
    X_train,
    y_train,
    epochs=250,
    batch_size=32,
    lr=4e-4,
):
    model = model.to(device)
    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 25 == 0:
            print(f"[classifier] epoch {epoch+1:03d} | loss={total_loss / len(ds):.6f}")
    return model

def compute_detector_threshold(
    classifier,
    detector,
    X_train,
    y_train,
    concepts,
    masks,
    batch_size=32,
    lambda_rec=0.1,
    lambda_rel=0.01,
    lambda_pre=0.001,
):

#det_thr = mean(lambda_rec*rec_loss+lambda_rel*rel_loss+lambda_pre*pre_loss)
    classifier.eval()
    detector.eval()
    ds = DreamDataset(X_train, y_train, concepts, masks)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    values = []
    with torch.no_grad():
        for xb, yb, cb, mb in dl:
            xb = xb.to(device)
            cb = cb.to(device)
            mb = mb.to(device)
            x_hat, z, concept_probs = detector(xb)
            rec = torch.mean((x_hat - xb) ** 2, dim=1)
            rel = concept_reliability_loss_per_sample(classifier, xb, x_hat)
            raw_pre = F.binary_cross_entropy(
                concept_probs.clamp(1e-7, 1.0 - 1e-7),
                cb,
                reduction="none",
            )
            pre = (raw_pre * mb).sum(dim=1) / mb.sum(dim=1).clamp_min(1.0)
            det_loss = lambda_rec * rec + lambda_rel * rel + lambda_pre * pre
            values.append(det_loss.detach().cpu().numpy())
    return float(np.concatenate(values).mean())

def train_dream_detector(
    detector,
    classifier,
    X_train,
    y_train,
    concepts,
    masks,
    epochs=250,
    batch_size=32,
    lr=4e-4,
    lambda_rec=0.1,
    lambda_sep=0.01,
    lambda_rel=0.01,
    lambda_pre=0.001,
    sep_margin=10.0,
):
    detector = detector.to(device)
    classifier = classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    ds = DreamDataset(X_train, y_train, concepts, masks)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(detector.parameters(), lr=lr)
    for epoch in range(epochs):
        detector.train()
        total = {
            "loss": 0.0,
            "rec": 0.0,
            "sep": 0.0,
            "rel": 0.0,
            "pre": 0.0,
        }
        for xb, yb, cb, mb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            cb = cb.to(device)
            mb = mb.to(device)
            x_hat, z, concept_probs = detector(xb)
            loss_rec = reconstruction_loss(x_hat, xb)
            loss_sep = cade_contrastive_separation_loss(z, yb, margin=sep_margin)
            loss_rel = concept_reliability_loss(classifier, xb, x_hat)
            loss_pre = concept_presence_loss(concept_probs, cb, mb)
            loss = (
                lambda_rec * loss_rec
                + lambda_sep * loss_sep
                + lambda_rel * loss_rel
                + lambda_pre * loss_pre
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 5.0)
            opt.step()
            bs = len(xb)
            total["loss"] += loss.item() * bs
            total["rec"] += loss_rec.item() * bs
            total["sep"] += loss_sep.item() * bs
            total["rel"] += loss_rel.item() * bs
            total["pre"] += loss_pre.item() * bs
        if (epoch + 1) % 25 == 0:
            n = len(ds)
            print(
                f"[detector] epoch {epoch+1:03d} | "
                f"loss={total['loss']/n:.6f} | "
                f"rec={total['rec']/n:.6f} | "
                f"sep={total['sep']/n:.6f} | "
                f"rel={total['rel']/n:.6f} | "
                f"pre={total['pre']/n:.6f}"
            )
    for p in classifier.parameters():
        p.requires_grad = True
    return detector

##drift score

@torch.no_grad()
def dream_drift_scores(
    classifier,
    detector,
    X_eval,
    batch_size=32,
    lambda_rel=0.01,
):
    classifier.eval()
    detector.eval()
    ds = TensorDataset(torch.tensor(X_eval, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores = []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = classifier(xb)
        pseudo_ce = pseudo_cross_entropy_per_sample(logits)
        x_hat, _, _ = detector(xb)
        rel = concept_reliability_loss_per_sample(classifier, xb, x_hat)
        score = pseudo_ce + lambda_rel * rel
        scores.append(score.detach().cpu().numpy())
    return np.concatenate(scores)

##top k

def select_top_k_by_score(scores, k):
    k = min(k, len(scores))
    return np.argsort(scores)[::-1][:k]

##concept space drift explainer

def compute_class_centroids(detector, X_train, y_train, batch_size=32):
    detector.eval()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    z_all = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            _, z, _ = detector(xb)
            z_all.append(z.cpu())

    z_all = torch.cat(z_all, dim=0)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    centroids = {}
    closest_samples = {}
    for cls in sorted(np.unique(y_train)):
        idx = torch.where(y_tensor == int(cls))[0]
        z_cls = z_all[idx]
        centroid = z_cls.mean(dim=0)
        dist = torch.sqrt(torch.sum((z_cls - centroid) ** 2, dim=1) + 1e-10)
        nearest_local = torch.argmin(dist).item()
        nearest_global = idx[nearest_local].item()
        centroids[int(cls)] = centroid.to(device)
        closest_samples[int(cls)] = X_train[nearest_global]
    return centroids, closest_samples

def gumbel_sigmoid_sample(p, batch_size, temp=0.1):
    eps = np.finfo(float).eps
    p = p.clamp(eps, 1.0 - eps)
    u = torch.rand((batch_size,) + tuple(p.shape), device=p.device).clamp(eps, 1.0 - eps)
    logit = (
        torch.log(p)
        - torch.log(1.0 - p)
        + torch.log(u)
        - torch.log(1.0 - u)
    ) / temp
    return torch.sigmoid(logit)

def explain_instance_concept_space(
    x_drift_np,
    classifier,
    detector,
    centroids,
    closest_samples,
    lambda_1=0.001,
    alpha_u=1.0,
    batch_size=10,
    exp_epochs=250,
    exp_lambda_patience=20,
    early_stop_patience=10,
    temp=0.1,
):
#return binary mask over latent concept dimensions, shape [encoding_dim]
#first 10 positions are explicit behavior concepts

    classifier.eval()
    detector.eval()
    x_drift = torch.tensor(x_drift_np[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_label = classifier(x_drift).argmax(dim=1).item()
        x_ref_np = closest_samples[pred_label]
        x_ref = torch.tensor(x_ref_np[None, :], dtype=torch.float32, device=device)
        _, z_drift, _ = detector(x_drift)
        _, z_ref, _ = detector(x_ref)
        c_y = centroids[pred_label].detach()
        ref_pred = F.softmax(classifier(x_ref), dim=1).detach()
    mask_shape = z_drift.shape[1:]
    p = torch.nn.Parameter(torch.rand(mask_shape, device=device))
    optimizer = torch.optim.Adam([p], lr=1e-2)
    best_mask = None
    best_loss = float("inf")
    no_improve = 0
    lambda_wait = 0
    active_lambda = lambda_1
    for epoch in range(exp_epochs):
        p_norm = torch.sigmoid(p)
        m = gumbel_sigmoid_sample(p_norm, batch_size=batch_size, temp=temp)
        z_d = z_drift.repeat(batch_size, 1)
        z_r = z_ref.repeat(batch_size, 1)
        z_prime = z_d * (1.0 - m) + z_r * m
        x_prime = detector.decoder(z_prime)
        z_prime_check = detector.encoder(x_prime)
        distance_loss = torch.sqrt(torch.sum((z_prime_check - c_y) ** 2, dim=1) + 1e-10).mean()
        logits_prime = classifier(x_prime)
        pseudo_ce = F.cross_entropy(
            logits_prime,
            logits_prime.argmax(dim=1),
            reduction="mean",
        )
        log_probs_prime = F.log_softmax(logits_prime, dim=1)
        rel_loss = -(ref_pred.repeat(batch_size, 1) * log_probs_prime).sum(dim=1).mean()
        uncertainty_loss = pseudo_ce + rel_loss
        regularization = torch.sum(torch.abs(m)) / batch_size
        l2_reg = torch.sqrt(torch.sum(m * m) + 1e-10) / batch_size
        loss = distance_loss + alpha_u * uncertainty_loss + active_lambda * (regularization + l2_reg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current = loss.item()
        if current < best_loss:
            best_loss = current
            best_mask = p_norm.detach().clone()
            no_improve = 0
            lambda_wait = 0
        else:
            no_improve += 1
            lambda_wait += 1

#if mask isnt sparse enough, gradually strengthen regularization.
        if lambda_wait >= exp_lambda_patience:
            active_lambda *= 2.0
            lambda_wait = 0
        if no_improve >= early_stop_patience:
            break
    if best_mask is None:
        best_mask = torch.sigmoid(p.detach())
    binary_mask = (best_mask > 0.5).float().cpu().numpy().astype(np.int64)
    return binary_mask

def explain_topk_drift_samples(
    classifier,
    detector,
    X_train,
    y_train,
    X_test,
    selected_idx,
):
    centroids, closest_samples = compute_class_centroids(detector, X_train, y_train)
    masks = []
    for idx in tqdm(selected_idx, desc="dream concept explanations"):
        mask = explain_instance_concept_space(
            X_test[idx],
            classifier,
            detector,
            centroids,
            closest_samples,
            lambda_1=0.001,
            alpha_u=1.0,
            batch_size=10,
            #testing!!
            exp_epochs=5,
            #exp_epochs=250,
            exp_lambda_patience=20,
            early_stop_patience=10,
            temp=0.1,
        )
        masks.append(mask)
    return np.array(masks)

##adaptation

def expand_krono_classifier(old_model, new_num_classes):
    old_model_cpu = copy.deepcopy(old_model).cpu()
    old_state = old_model_cpu.state_dict()
    input_dim = old_model_cpu.feature[0].in_features
    new_model = DrebinMLP(input_dim, new_num_classes)
    new_state = new_model.state_dict()
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
#copy final layer partially
    old_w = old_state["out.weight"]
    old_b = old_state["out.bias"]
    new_state["out.weight"][: old_w.shape[0]] = old_w
    new_state["out.bias"][: old_b.shape[0]] = old_b
    new_model.load_state_dict(new_state)
    return new_model

def dream_joint_adaptation_with_schedule(
    old_classifier,
    old_detector,
    X_train,
    y_train,
    concept_train,
    mask_train,
    X_test,
    y_test_for_adapt,
    concept_test,
    mask_test,
    selected_idx,
    num_total_classes,
    epochs=250,
    batch_size=32,
    lr=4e-4,
    lambda_rec=0.1,
    lambda_sep=0.01,
    lambda_rel=0.01,
    lambda_pre=0.001,
    sep_margin=10.0,
    eta=0.1,
):
    classifier = expand_krono_classifier(old_classifier, num_total_classes).to(device)
    detector = copy.deepcopy(old_detector).to(device)
    X_selected = X_test[selected_idx]
    y_selected = y_test_for_adapt[selected_idx]
    c_selected = concept_test[selected_idx]
    m_selected = mask_test[selected_idx]
    X_adapt = np.concatenate([X_train, X_selected], axis=0)
    y_adapt = np.concatenate([y_train, y_selected], axis=0)
    c_adapt = np.concatenate([concept_train, c_selected], axis=0)
    m_adapt = np.concatenate([mask_train, m_selected], axis=0)

#threshold from training data before adaptaion
    det_thr = compute_detector_threshold(
        classifier=old_classifier.to(device),
        detector=old_detector.to(device),
        X_train=X_train,
        y_train=y_train,
        concepts=concept_train,
        masks=mask_train,
        batch_size=batch_size,
        lambda_rec=lambda_rec,
        lambda_rel=lambda_rel,
        lambda_pre=lambda_pre,
    )
    print(f"[adapt] detector threshold = {det_thr:.8f}")
    print(f"[adapt] selected samples = {len(selected_idx)}")
    ds = DreamDataset(X_adapt, y_adapt, c_adapt, m_adapt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    detector_optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
    detector_lr_reduced = False
    for epoch in range(epochs):
        classifier.train()
        detector.train()
        total_loss = 0.0
        total_det_loss = 0.0
        for xb, yb, cb, mb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            cb = cb.to(device)
            mb = mb.to(device)
            logits = classifier(xb)
            cls_loss = F.cross_entropy(logits, yb)
            x_hat, z, concept_probs = detector(xb)
            rec_loss = reconstruction_loss(x_hat, xb)
            sep_loss = cade_contrastive_separation_loss(z, yb, margin=sep_margin)
            rel_loss = concept_reliability_loss(classifier, xb, x_hat)
            pre_loss = concept_presence_loss(concept_probs, cb, mb)
            det_loss = (
                lambda_rec * rec_loss
                + lambda_rel * rel_loss
                + lambda_pre * pre_loss
            )
            total = (
                cls_loss
                + lambda_rec * rec_loss
                + lambda_sep * sep_loss
                + lambda_rel * rel_loss
                + lambda_pre * pre_loss
            )
            classifier_optimizer.zero_grad()
            detector_optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 5.0)
            classifier_optimizer.step()

#dynamic detector learning rate schedule
            if (not detector_lr_reduced) and det_loss.item() < det_thr:
                for group in detector_optimizer.param_groups:
                    group["lr"] = group["lr"] * eta
                detector_lr_reduced = True
                print(
                    f"[adapt] detector lr reduced at epoch {epoch+1}: "
                    f"{lr} -> {lr * eta}"
                )
            detector_optimizer.step()
            total_loss += total.item() * len(xb)
            total_det_loss += det_loss.item() * len(xb)
        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            n = len(ds)
            current_det_lr = detector_optimizer.param_groups[0]["lr"]
            print(
                f"[joint adapt] epoch {epoch+1:03d} | "
                f"loss={total_loss/n:.6f} | "
                f"det_loss={total_det_loss/n:.6f} | "
                f"det_lr={current_det_lr:.8f}"
            )
    return classifier, detector

@torch.no_grad()
def predict_labels(model, X_eval, batch_size=32):
    model.eval()
    ds = TensorDataset(torch.tensor(X_eval, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []

    for (xb,) in dl:
        xb = xb.to(device)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)

def evaluate_classifier(model, X_eval, y_true):
    y_pred = predict_labels(model, X_eval)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

#run one holdout family
holdout_family = sorted(np.unique(y))[0]
split = make_holdout_split(
    X,
    y,
    concepts,
    concept_mask,
    holdout_family=holdout_family,
    test_size=0.2,
    seed=OFFICIAL["seed"],
)
print("Holdout family:", holdout_family, family_names[holdout_family])
print("Train:", split["X_train"].shape)
print("Test:", split["X_test"].shape)
print("Known classes:", split["num_known_classes"])
input_dim = split["X_train"].shape[1]
num_known_classes = split["num_known_classes"]
classifier = DrebinMLP(
    input_dim=input_dim,
    num_classes=num_known_classes,
)
classifier = train_classifier(
    classifier,
    split["X_train"],
    split["y_train"],
    epochs=OFFICIAL["classifier_epochs"],
    batch_size=OFFICIAL["batch_size"],
    lr=OFFICIAL["learning_rate"],
)
detector = DreamDetector(
    input_dim=input_dim,
    num_concepts=len(CONCEPT_NAMES),
    hidden_dim=OFFICIAL["ae_hidden_dim"],
    encoding_dim=OFFICIAL["ae_encoding_dim"],
)
detector = train_dream_detector(
    detector,
    classifier,
    split["X_train"],
    split["y_train"],
    split["concept_train"],
    split["mask_train"],
    epochs=OFFICIAL["detector_epochs"],
    batch_size=OFFICIAL["batch_size"],
    lr=OFFICIAL["learning_rate"],
    lambda_rec=OFFICIAL["lambda_rec"],
    lambda_sep=OFFICIAL["lambda_sep"],
    lambda_rel=OFFICIAL["lambda_rel"],
    lambda_pre=OFFICIAL["lambda_pre"],
    sep_margin=OFFICIAL["sep_margin"],
)
scores = dream_drift_scores(
    classifier,
    detector,
    split["X_test"],
    batch_size=OFFICIAL["batch_size"],
    lambda_rel=OFFICIAL["lambda_rel"],
)
auc = roc_auc_score(split["drift_binary"], scores)
print("Drift detection AUC:", auc)
budget = 20
selected_idx = select_top_k_by_score(scores, budget)
print("Selected:", len(selected_idx))
print("True drift selected:", int(split["drift_binary"][selected_idx].sum()), "/", budget)
#concept space explanation for selected drift sample
explanation_masks = explain_topk_drift_samples(
    classifier,
    detector,
    split["X_train"],
    split["y_train"],
    split["X_test"],
    selected_idx,
)
print("Explanation masks:", explanation_masks.shape)
print("Explicit concept masks first sample:", explanation_masks[0][:10])
print("Explicit concepts:", [CONCEPT_NAMES[i] for i in np.where(explanation_masks[0][:10] == 1)[0]])
adapted_classifier, adapted_detector = dream_joint_adaptation_with_schedule(
    classifier,
    detector,
    split["X_train"],
    split["y_train"],
    split["concept_train"],
    split["mask_train"],
    split["X_test"],
    split["y_test_for_adapt"],
    split["concept_test"],
    split["mask_test"],
    selected_idx,
    num_total_classes=split["num_known_classes"] + 1,
    epochs=OFFICIAL["adapt_epochs"],
    batch_size=OFFICIAL["batch_size"],
    lr=OFFICIAL["learning_rate"],
    lambda_rec=OFFICIAL["lambda_rec"],
    lambda_sep=OFFICIAL["lambda_sep"],
    lambda_rel=OFFICIAL["lambda_rel"],
    lambda_pre=OFFICIAL["lambda_pre"],
    sep_margin=OFFICIAL["sep_margin"],
    eta=OFFICIAL["eta"],
)
metrics = evaluate_classifier(
    adapted_classifier,
    split["X_test"],
    split["y_test_for_adapt"],
)
print("Dream adapted classifier:", metrics)

##save

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
torch.save(
    {
        "classifier": classifier.state_dict(),
        "detector": detector.state_dict(),
        "adapted_classifier": adapted_classifier.state_dict(),
        "adapted_detector": adapted_detector.state_dict(),
        "input_dim": input_dim,
        "num_known_classes": num_known_classes,
        "concept_names": CONCEPT_NAMES,
        "official": OFFICIAL,
        "holdout_family": int(holdout_family),
    },
    "checkpoints/dream_krono_official_like_pytorch.pt",
)
np.savez(
    "results/one_holdout_results.npz",
    scores=scores,
    drift_binary=split["drift_binary"],
    selected_idx=selected_idx,
    explanation_masks=explanation_masks,
    accuracy=metrics["accuracy"],
    f1_macro=metrics["f1_macro"],
    auc=auc,
)