import os
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DATA_DIR = Path("data") / "plant_multi"
TRAIN_DIR = DATA_DIR / "train"

IMG_SIZE = (224, 224)

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Filenames
RF_NAME = OUT_DIR / "rf_glcm_multi.pkl"
SVM_NAME = OUT_DIR / "svm_glcm_multi.pkl"
KNN_NAME = OUT_DIR / "knn_glcm_multi.pkl"
CLASSNAMES_NAME = OUT_DIR / "class_names_multi.pkl"

# If you have precomputed CNN embeddings (penultimate-layer features), save them as:
# np.savez("models/embeddings.npz", embeddings=emb_array, labels=labels_array, class_names=class_names_list)
EMBEDDINGS_PATH = OUT_DIR / "embeddings.npz"  # optional

# GLCM props (kept same as your original)
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


# ------------ GLCM FEATURE EXTRACTION ------------

def extract_glcm_features_from_path(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_q = (gray // 32).astype(np.uint8)  # 8 levels: 0..7

    glcm = graycomatrix(gray_q, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)

    feats = []
    for prop in GLCM_PROPS:
        feats.append(graycoprops(glcm, prop)[0, 0])

    return np.array(feats, dtype=np.float32)


# ------------ BUILD DATASET FROM FOLDERS (GLCM) ------------

def build_glcm_dataset_from_folder(root_dir):
    X = []
    y = []
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print("Class mapping:", class_to_idx)

    for cls_name in class_names:
        cls_path = os.path.join(root_dir, cls_name)
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            if not os.path.isfile(fpath):
                continue
            feats = extract_glcm_features_from_path(fpath)
            if feats is None:
                continue
            X.append(feats)
            y.append(class_to_idx[cls_name])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print("GLCM Features shape:", X.shape)
    print("Num samples:", len(y))
    return X, y, class_names


# ------------ LOAD EMBEDDINGS (OPTIONAL) ------------

def load_embeddings_npz(path):
    data = np.load(path, allow_pickle=True)
    # Expect keys: 'embeddings', 'labels', 'class_names' (class_names optional)
    emb = data.get("embeddings", None)
    labels = data.get("labels", None)
    class_names = data.get("class_names", None)
    if class_names is not None:
        # ensure python list of strings
        class_names = [c.decode("utf-8") if isinstance(c, bytes) else c for c in class_names]
    return emb, labels, class_names


# ------------ TRAIN / SAVE CLASSICAL MODELS ------------

def make_pipelines(random_state=42):
    rf = Pipeline([("scaler", StandardScaler()),
                   ("rf", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1))])
    svm = Pipeline([("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", probability=True, random_state=random_state))])
    knn = Pipeline([("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])
    return rf, svm, knn


def train_and_save_models(X, y, class_names):
    print("Creating pipelines and training models...")
    rf, svm, knn = make_pipelines()

    print("\nFitting Random Forest...")
    rf.fit(X, y)
    print("Random Forest trained.")

    print("\nFitting SVM...")
    svm.fit(X, y)
    print("SVM trained.")

    print("\nFitting KNN...")
    knn.fit(X, y)
    print("KNN trained.")

    # Save artifacts
    joblib.dump(rf, RF_NAME)
    joblib.dump(svm, SVM_NAME)
    joblib.dump(knn, KNN_NAME)
    joblib.dump(class_names, CLASSNAMES_NAME)

    print(f"\nSaved models:\n - {RF_NAME}\n - {SVM_NAME}\n - {KNN_NAME}\n - {CLASSNAMES_NAME}")


# ------------ MAIN ------------

def main():
    print("=== Classical model trainer ===")
    # Prefer embeddings if available (useful after you re-train CNNs)
    if EMBEDDINGS_PATH.exists():
        try:
            print(f"[INFO] Found embeddings file {EMBEDDINGS_PATH}. Using embeddings for training.")
            emb, labels, cn = load_embeddings_npz(str(EMBEDDINGS_PATH))
            if emb is None or labels is None:
                raise ValueError("embeddings.npz missing 'embeddings' or 'labels' arrays.")
            X = np.array(emb, dtype=np.float32)
            y = np.array(labels, dtype=np.int64)
            if cn is None:
                # if no class_names in embeddings file, try to infer from folders
                _, _, cn = build_glcm_dataset_from_folder(TRAIN_DIR)
            train_and_save_models(X, y, cn)
            return
        except Exception as e:
            print("[WARN] Failed to load/use embeddings.npz:", e)
            print("[WARN] Falling back to GLCM features.")

    # Fall back to GLCM features from images
    print(f"[INFO] Building GLCM dataset from {TRAIN_DIR}")
    X, y, class_names = build_glcm_dataset_from_folder(str(TRAIN_DIR))
    if X.size == 0:
        raise RuntimeError("No GLCM features extracted. Check TRAIN_DIR and files.")
    train_and_save_models(X, y, class_names)


if __name__ == "__main__":
    main()
