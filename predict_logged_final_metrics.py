
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import roc_curve, auc

model_paths = ['densenet_model.keras', 'inceptionv3_model.keras', 'xception_model.keras']
model_names = ['DenseNet121', 'InceptionV3', 'Xception']
models = [load_model(p) for p in model_paths]
image_size = (300, 300)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def generate_grad_cam(model, img_array, last_conv_layer_name='block14_sepconv2'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 1]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap + 1e-8)
    return heatmap

def predict_image(img_path, generate_gradcam=True):
    img_array = preprocess_image(img_path)
    all_probs = []
    for model, name in zip(models, model_names):
        probs = model.predict(img_array, verbose=0)[0]
        all_probs.append(probs)
    ensemble_probs = np.mean(all_probs, axis=0)
    predicted_label = int(np.argmax(ensemble_probs))

    if generate_gradcam:
        print("Grad-CAM для Xception")
        heatmap = generate_grad_cam(models[2], img_array)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        heatmap = cv2.resize(heatmap, image_size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        output_path = "gradcam_output.jpg"
        cv2.imwrite(output_path, superimposed_img)
        print(f"Grad-CAM сохранён: {output_path}")

    return predicted_label

y_true_all = []
y_score_all = []

def evaluate_folder(real_dir, fake_dir):
    correct = 0
    total = 0

    for img_name in os.listdir(real_dir):
        img_path = os.path.join(real_dir, img_name)
        pred = predict_image(img_path)
        score = np.mean([model.predict(preprocess_image(img_path), verbose=0)[0][1] for model in models])
        y_true_all.append(0)
        y_score_all.append(1 - score)
        if pred == 0:
            correct += 1
        total += 1

    for img_name in os.listdir(fake_dir):
        img_path = os.path.join(fake_dir, img_name)
        pred = predict_image(img_path)
        score = np.mean([model.predict(preprocess_image(img_path), verbose=0)[0][1] for model in models])
        y_true_all.append(1)
        y_score_all.append(score)
        if pred == 1:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Точность ансамбля на {os.path.basename(real_dir).split('/')[0]}: {accuracy:.4f} ({correct}/{total})")

# Пример
test_image_path = '/Users/victorvasss/Documents/polytech/diploma/research/dataset/SNGAN/0_real/000008.png'
print(f"Предсказал: {predict_image(test_image_path)}")

# evaluate_folder(
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/CramerGAN/0_real',
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/CramerGAN/1_fake'
# )
# evaluate_folder(
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/STGAN/0_real',
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/STGAN/1_fake'
# )
# evaluate_folder(
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/AttGAN/0_real',
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/AttGAN/1_fake'
# )
# evaluate_folder(
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/S3GAN/0_real',
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/S3GAN/1_fake'
# )
# evaluate_folder(
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/SNGAN/0_real',
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/SNGAN/1_fake'
# )
# evaluate_folder(
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/MMDGAN/0_real',
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/MMDGAN/1_fake'
# )
# evaluate_folder(
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/BEGAN/0_real',
#     '/Users/victorvasss/Documents/polytech/diploma/research/dataset/BEGAN/1_fake'
# )
#
# fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='Ensemble ROC curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC-кривая ансамбля')
# plt.legend(loc="lower right")
# plt.savefig("ensemble_roc_curve.png")
# print("ROC-кривая ансамбля сохранена: ensemble_roc_curve.png")
#
#
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
#
# cm = confusion_matrix(y_true_all, [1 if s >= 0.5 else 0 for s in y_score_all])
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix - Ensemble')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.savefig("ensemble_confusion_matrix.png")
# print("Матрица ошибок ансамбля сохранена: ensemble_confusion_matrix.png")
#
#
# from sklearn.metrics import classification_report
# print("\nКлассификационный отчёт по ансамблю:")
# print(classification_report(y_true_all, [1 if s >= 0.5 else 0 for s in y_score_all], digits=4))
#
#
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Перевод вероятностей в метки по порогу
# y_pred_all = [1 if s >= 0.5 else 0 for s in y_score_all]
#
# # classification report
# print("\nКлассификационный отчёт по ансамблю (по меткам):")
# print(classification_report(y_true_all, y_pred_all, digits=4))
#
# # Confusion Matrix
# cm = confusion_matrix(y_true_all, y_pred_all)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix - Ensemble')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.savefig("ensemble_confusion_matrix.png")
# print("Матрица ошибок ансамбля сохранена: ensemble_confusion_matrix.png")
#
# # ROC-кривая (остается по вероятностям)
# fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='Ensemble ROC curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC-кривая ансамбля')
# plt.legend(loc="lower right")
# plt.savefig("ensemble_roc_curve.png")
# print("ROC-кривая ансамбля сохранена: ensemble_roc_curve.png")
