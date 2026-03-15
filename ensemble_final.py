
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.applications import DenseNet121, InceptionV3, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# Load metadata
with open('train_dataset.pkl', 'rb') as ft:
    train_data = pickle.load(ft)
with open('val_dataset.pkl', 'rb') as fv:
    val_data = pickle.load(fv)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32

def decode_img(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.one_hot(label, depth=2)
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_data['image_path'].tolist(), train_data['label'].tolist()))
val_ds = tf.data.Dataset.from_tensor_slices((val_data['image_path'].tolist(), val_data['label'].tolist()))

train_ds = train_ds.map(decode_img).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(decode_img).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

models = [
    (DenseNet121, 'densenet_model.keras'),
    (InceptionV3, 'inceptionv3_model.keras'),
    (Xception, 'xception_model.keras')
]

for model_class, model_path in models:
    print(f"Обучение модели: {model_class.__name__}")

    base_model = model_class(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    for layer in base_model.layers[-50:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              callbacks=[ModelCheckpoint(model_path, save_best_only=True)])

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(preds, axis=1).numpy())
        y_prob.extend(preds[:, 1])

    print(f"Classification Report for {model_class.__name__}:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_class.__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{model_class.__name__}.png')
    plt.clf()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_class.__name__} (AUC = {roc_auc:.2f})')

    del model, y_true, y_pred, y_prob
    gc.collect()

plt.title("ROC Curve - All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("roc_curve_all_models.png")
print("Все метрики и модели сохранены.")
