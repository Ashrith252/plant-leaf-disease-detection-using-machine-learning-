import os
import time
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
import tensorflow as tf

from skimage.feature import graycomatrix, graycoprops

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
W1, W2, W3 = 0.55, 0.25, 0.20

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# candidate model filenames (edit if yours differ)
CNN_RGB_CANDIDATES = ["models/cnn_rgb_multi.h5", "models/cnn_rgb_tomato.h5", "models/cnn_rgb.h5"]
CNN_ENH_CANDIDATES = ["models/cnn_enhanced_multi.h5", "models/cnn_enhanced_tomato.h5", "models/cnn_enhanced.h5"]
RF_CANDIDATES = ["models/rf_glcm_multi.pkl", "models/rf_glcm_tomato.pkl", "models/rf_glcm.pkl"]
SVM_CANDIDATES = ["models/svm_glcm_multi.pkl", "models/svm_glcm_tomato.pkl", "models/svm_glcm.pkl"]
KNN_CANDIDATES = ["models/knn_glcm_multi.pkl", "models/knn_glcm_tomato.pkl", "models/knn_glcm.pkl"]
CLASSNAMES_CANDIDATES = ["models/class_names_multi.pkl", "models/class_names_tomato.pkl", "models/class_names.pkl"]

GLCM_FEATURE_NAMES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

# ---------------- Helpers ----------------

def try_load_path(candidates, loader):
    for p in candidates:
        if os.path.exists(p):
            try:
                return loader(p), p
            except Exception as e:
                print(f"[WARN] Found {p} but failed to load: {e}")
    return None, None

def unwrap_estimator(clf):
    """If clf is a Pipeline, return final estimator; else return clf."""
    try:
        if hasattr(clf, "named_steps"):
            last = list(clf.named_steps.keys())[-1]
            return clf.named_steps[last]
    except Exception:
        pass
    return clf

def load_models():
    cnn_rgb, p1 = try_load_path(CNN_RGB_CANDIDATES, load_model)
    cnn_enh, p2 = try_load_path(CNN_ENH_CANDIDATES, load_model)
    rf_clf, pr = try_load_path(RF_CANDIDATES, joblib.load)
    svm_clf, ps = try_load_path(SVM_CANDIDATES, joblib.load)
    knn_clf, pk = try_load_path(KNN_CANDIDATES, joblib.load)
    class_names, pc = try_load_path(CLASSNAMES_CANDIDATES, joblib.load)

    if not all([cnn_rgb, cnn_enh, rf_clf, svm_clf, knn_clf, class_names]):
        missing = []
        if not cnn_rgb: missing.append("CNN_RGB")
        if not cnn_enh: missing.append("CNN_ENH")
        if not rf_clf: missing.append("RF")
        if not svm_clf: missing.append("SVM")
        if not knn_clf: missing.append("KNN")
        if not class_names: missing.append("CLASS_NAMES")
        raise FileNotFoundError("Missing models: " + ",".join(missing))

    # ensure class_names are strings
    class_names = [c.decode("utf-8") if isinstance(c, bytes) else c for c in class_names]

    print(f"[INFO] Loaded models:\n CNN_RGB: {p1}\n CNN_ENH: {p2}\n RF: {pr}\n SVM: {ps}\n KNN: {pk}\n class_names: {pc}")
    return cnn_rgb, cnn_enh, rf_clf, svm_clf, knn_clf, class_names

def preprocess_for_cnn(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    return img_resized.astype(np.float32) / 255.0

def apply_clahe_to_single(rgb_img):
    img_uint8 = (rgb_img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced.astype(np.float32) / 255.0

def extract_glcm_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_q = (gray // 32).astype(np.uint8)   # 8 levels
    glcm = graycomatrix(gray_q, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
    feats = [graycoprops(glcm, p)[0,0] for p in GLCM_FEATURE_NAMES]
    return np.array(feats, dtype=np.float32).reshape(1, -1)

def parse_class_name(full_name):
    if isinstance(full_name, bytes):
        full_name = full_name.decode("utf-8")
    if "___" in full_name:
        plant, disease_raw = full_name.split("___", 1)
    else:
        parts = full_name.split("_")
        if parts and parts[0].lower() in ("tomato","potato"):
            plant = parts[0].capitalize()
            disease_raw = "_".join(parts[1:]) if len(parts)>1 else "unknown"
        else:
            plant, disease_raw = "Unknown", full_name
    return plant, disease_raw.replace("_"," ")

def interpret_bligh_type(pred_class):
    cl = pred_class.lower()
    if "early_blight".replace("_","") in cl or "early blight" in cl:
        return "EARLY BLIGHT"
    if "late_blight".replace("_","") in cl or "late blight" in cl:
        return "LATE BLIGHT"
    if "yellow" in cl and "curl" in cl:
        return "YELLOW LEAF CURL VIRUS"
    return None

# ----- visualization helpers -----

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        name = getattr(layer, "name", "").lower()
        outshape = getattr(layer, "output_shape", None)
        if "conv" in name and outshape is not None and len(outshape) == 4:
            return layer.name
    for layer in reversed(model.layers):
        outshape = getattr(layer, "output_shape", None)
        if outshape is not None and len(outshape) == 4:
            return layer.name
    raise ValueError("No conv layer found in model.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:,:,i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        heatmap = np.zeros_like(heatmap)
    else:
        heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    return heatmap

def overlay_heatmap_on_image(img_rgb_uint8, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    overlay = cv2.addWeighted(colored, alpha, img_rgb_uint8, 1 - alpha, 0)
    return overlay

def save_bar_confidences(conf_rgb, conf_clahe, conf_classical, conf_final, outpath):
    labels = ["CNN RGB", "CNN CLAHE", "Classical Avg", "Ensemble"]
    vals = [conf_rgb, conf_clahe, conf_classical, conf_final]
    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.title("Expert Confidences")
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_rf_feature_importance(rf_clf, outpath):
    try:
        est = unwrap_estimator(rf_clf)
        fi = getattr(est, "feature_importances_", None)
        if fi is None:
            raise AttributeError("feature_importances_ not found on RF estimator.")
        fi = np.array(fi)
        n = min(len(fi), len(GLCM_FEATURE_NAMES))
        y = fi[:n]
        labels = GLCM_FEATURE_NAMES[:n]
        plt.figure(figsize=(6,4))
        bars = plt.bar(labels, y)
        plt.title("Random Forest - GLCM Feature Importances")
        plt.ylabel("Importance")
        for bar, v in zip(bars, y):
            plt.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}", ha='center')
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return outpath
    except Exception as e:
        raise RuntimeError(f"RF feature importance failed: {e}")

# ---------------- PREDICT + VISUALIZE ----------------

def predict_image_and_visualize(image_path, cnn_rgb, cnn_enh, rf_clf, svm_clf, knn_clf, class_names):
    ts = int(time.time())
    img_rgb = preprocess_for_cnn(image_path)   # float 0-1
    enh = apply_clahe_to_single(img_rgb)

    rgb_batch = np.expand_dims(img_rgb, 0)
    enh_batch = np.expand_dims(enh, 0)

    # CNN predictions
    p1 = cnn_rgb.predict(rgb_batch, verbose=0)[0]
    p2 = cnn_enh.predict(enh_batch, verbose=0)[0]

    idx1 = int(np.argmax(p1)); class1 = class_names[idx1]; conf1 = float(p1[idx1])
    idx2 = int(np.argmax(p2)); class2 = class_names[idx2]; conf2 = float(p2[idx2])

    # derive plant/disease for each expert
    plant1, disease1 = parse_class_name(class1)
    plant2, disease2 = parse_class_name(class2)

    # classical
    glcm = extract_glcm_features(image_path)
    p_rf = rf_clf.predict_proba(glcm)[0]
    p_svm = svm_clf.predict_proba(glcm)[0]
    p_knn = knn_clf.predict_proba(glcm)[0]
    p_classical = (p_rf + p_svm + p_knn) / 3.0
    idx3 = int(np.argmax(p_classical)); class3 = class_names[idx3]; conf3 = float(p_classical[idx3])
    plant3, disease3 = parse_class_name(class3)

    # final fusion
    p_final = W1*p1 + W2*p2 + W3*p_classical
    idxf = int(np.argmax(p_final)); classf = class_names[idxf]; conff = float(p_final[idxf])
    plantf, diseasef = parse_class_name(classf)

    # verdict block (corrected)
    blight = interpret_bligh_type(classf)
    if blight is not None:
        verdict = blight
    elif "healthy" in classf.lower() or "healthy" in diseasef.lower():
        verdict = "HEALTHY"
    else:
        if diseasef and diseasef.strip() and diseasef.lower() != "unknown":
            verdict = diseasef.upper()
        else:
            verdict = "OTHER DISEASE / UNKNOWN"

    # print summary (enhanced with plant info per expert)
    print("\n====== PREDICTION SUMMARY ======")
    print(f"Expert1 (CNN RGB)       : {class1}  ({conf1:.4f})")
    print(f"   => Plant: {plant1} | Disease (pretty): {disease1}")
    print(f"Expert2 (CNN CLAHE)     : {class2}  ({conf2:.4f})")
    print(f"   => Plant: {plant2} | Disease (pretty): {disease2}")
    print(f"Expert3 (Classical avg) : {class3}  ({conf3:.4f})")
    print(f"   => Plant: {plant3} | Disease (pretty): {disease3}")
    print("-----")
    print(f"FINAL                   : {classf}  ({conff:.4f})")
    print(f"   => Plant: {plantf} | Disease (pretty): {diseasef}")
    print(f"Verdict                 : {verdict}")
    print("================================\n")

    outputs = {}

    # save bar chart
    try:
        barpath = os.path.join(OUTPUT_DIR, f"confidence_bar_{ts}.png")
        save_bar_confidences(conf1, conf2, conf3, conff, barpath)
        outputs['bar'] = barpath
    except Exception as e:
        print("[WARN] save bar failed:", e)

    # Grad-CAM for CNN RGB (explain final ensemble class)
    try:
        last_conv_rgb = find_last_conv_layer(cnn_rgb)
        heatmap_rgb = make_gradcam_heatmap(rgb_batch, cnn_rgb, last_conv_rgb, pred_index=idxf)
        img_uint8 = (img_rgb*255).astype(np.uint8)
        overlay_rgb = overlay_heatmap_on_image(img_uint8, heatmap_rgb)
        gpath_rgb = os.path.join(OUTPUT_DIR, f"gradcam_cnn_rgb_{ts}.png")
        cv2.imwrite(gpath_rgb, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        outputs['gradcam_rgb'] = gpath_rgb
    except Exception as e:
        print("[WARN] Grad-CAM failed for CNN RGB:", e)

    # Grad-CAM for CNN ENH
    try:
        last_conv_enh = find_last_conv_layer(cnn_enh)
        heatmap_enh = make_gradcam_heatmap(enh_batch, cnn_enh, last_conv_enh, pred_index=idxf)
        img_uint8 = (img_rgb*255).astype(np.uint8)
        overlay_enh = overlay_heatmap_on_image(img_uint8, heatmap_enh)
        gpath_enh = os.path.join(OUTPUT_DIR, f"gradcam_cnn_enh_{ts}.png")
        cv2.imwrite(gpath_enh, cv2.cvtColor(overlay_enh, cv2.COLOR_RGB2BGR))
        outputs['gradcam_enh'] = gpath_enh
    except Exception as e:
        print("[WARN] Grad-CAM failed for CNN ENH:", e)

    # RF feature importance
    try:
        rf_path = os.path.join(OUTPUT_DIR, f"rf_feature_importance_{ts}.png")
        save_rf_feature_importance(rf_clf, rf_path)
        outputs['rf_feature_importance'] = rf_path
    except Exception as e:
        print("[WARN] RF feature importance failed:", e)

    return {
        "expert1": {"label": class1, "plant": plant1, "disease": disease1, "conf": conf1},
        "expert2": {"label": class2, "plant": plant2, "disease": disease2, "conf": conf2},
        "expert3": {"label": class3, "plant": plant3, "disease": disease3, "conf": conf3},
        "final": {"label": classf, "plant": plantf, "disease": diseasef, "conf": conff},
        "verdict": verdict,
        "files": outputs
    }

# ---------------- CLI ----------------

def main():
    try:
        cnn_rgb, cnn_enh, rf_clf, svm_clf, knn_clf, class_names = load_models()
    except Exception as e:
        print("[ERROR] Failed to load models:", e)
        return

    print("\nPlant / disease classifier (Grad-CAM + bar + RF importance) ready.")
    print("Type path to an image from your dataset (or absolute path). Type 'q' to quit.\n")

    while True:
        img_path = input("Image path: ").strip()
        if img_path.lower() in ("q","quit","exit"):
            print("Goodbye.")
            break
        if not os.path.isfile(img_path):
            print("File not found - try again (no quotes).")
            continue
        try:
            out = predict_image_and_visualize(img_path, cnn_rgb, cnn_enh, rf_clf, svm_clf, knn_clf, class_names)
            print("Saved visualization files:")
            for k,v in out['files'].items():
                print(f"  {k} -> {v}")
            print("\nOpen those images for your report / PPT.\n")
        except Exception as e:
            print("Error during prediction/visualization:", e)

if __name__ == "__main__":
    main()
