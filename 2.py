import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
import tensorflow as tf

# Tải dữ liệu
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)
# print("Kích thước bộ dữ liệu:")
# print(f"Kích thước train: {X_train.shape}, {y_train.shape}")
# print(f"Kích thước test: {X_test.shape}, {y_test.shape}")

# # In ra vài mẫu
# fig, axes = plt.subplots(10, 10, figsize=(10, 20))  # 10 hàng (0-9), mỗi hàng 10 ảnh

# for digit in range(10):
#     # Lấy index của các mẫu có nhãn = digit
#     idx = np.where(y_train == digit)[0][:10]  # lấy 10 mẫu đầu tiên

#     for i, ax in enumerate(axes[digit]):
#         ax.imshow(X_train[idx[i]].reshape(28, 28), cmap='gray')
#         ax.set_title(f"{digit}")
#         ax.axis('off')
# plt.tight_layout()
# plt.show()



# Kiểm tra phân bố nhãn
from collections import Counter

# Đếm tần suất xuất hiện của từng nhãn (0-9)
counter = Counter(y_train)  # Đếm số lần xuất hiện mỗi chữ số
counter = (
    counter.most_common()
)  # Trả về list các cặp (số, số lượng) sắp xếp theo tần suất
counter = pd.DataFrame(counter)  # Đưa vào DataFrame cho dễ xem
counter.columns = ["Số", "Số lượng"]  # Đặt tên cột

display(counter)  # Hiển thị bảng: mỗi số (0-9) có bao nhiêu ảnh

# Vẽ biểu đồ phân bố nhãn
# Sắp xếp theo cột 'Số' (tăng dần từ 0 -> 9)
counter_sorted = counter.sort_values(by="Số")

# Vẽ biểu đồ cột
plt.figure()
plt.bar(counter_sorted["Số"], counter_sorted["Số lượng"])

# Nhãn và tiêu đề
plt.xlabel("Chữ số")

# Ép hiển thị đầy đủ 0 -> 9 trên trục X
plt.xticks(range(10))
plt.ylabel("Số lượng ảnh")
plt.title("Phân phối nhãn trong MNIST")
plt.show()


from tensorflow.keras.utils import to_categorical
# ===== Preprocess cho CNN =====
def preprocess_for_cnn(X_train, X_test, y_train, y_test):
    # reshape về (28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # one-hot label
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, X_test, y_train, y_test


from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)

# Build model
model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(8, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(10)
])

# Compile model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Summary
model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",          # theo dõi validation loss
    patience=3,                  # không cải thiện 3 epoch thì dừng
    restore_best_weights=True    # quay lại model tốt nhất
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

# Đánh giá mô hình
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def pre_and_eva(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Nếu output là xác suất (CNN) -> convert sang label
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Nếu y_test là one-hot -> convert
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_test, y_pred) * 100
    cfs = confusion_matrix(y_test, y_pred)
    clr = classification_report(y_test, y_pred)

    print(f"Độ chính xác: {acc:.4f}%")
    print("-" * 50)
    print("Ma trận nhầm lẫn:")
    print(cfs)
    print("-" * 50)
    print("Classification report:")
    print(clr)

# Gọi hàm đánh giá cho mô hình KNN
pre_and_eva(model, X_test, y_test)

# Phân tích 10 mẫu dự đoán sai của mỗi số
import matplotlib.pyplot as plt
import numpy as np

# Dự đoán xác suất
y_pred_prob = model.predict(X_test)

# Convert sang label
y_pred = np.argmax(y_pred_prob, axis=1)

# Nếu y_test là one-hot -> convert
if len(y_test.shape) > 1:
    y_true = np.argmax(y_test, axis=1)
else:
    y_true = y_test

# Lấy index các mẫu dự đoán sai
wrong_idx = np.where(y_pred != y_true)[0]

# ===== VẼ =====
fig, axes = plt.subplots(10, 10, figsize=(15, 15))

for digit in range(10):
    # Lọc: label thật = digit nhưng dự đoán sai
    idx = wrong_idx[y_true[wrong_idx] == digit][:10]

    for i in range(10):
        ax = axes[digit, i]

        if i < len(idx):
            img_idx = idx[i]
            ax.imshow(X_test[img_idx].reshape(28, 28), cmap="gray")
            ax.set_title(f"T:{y_true[img_idx]} / P:{y_pred[img_idx]}")
        else:
            ax.axis("off")

        ax.axis("off")

plt.tight_layout()
plt.show()

# Tiền xử lý trước UI
from PIL import Image, ImageOps

def input_image(img_array):
    if img_array is None:
        return None

    if isinstance(img_array, dict):
        img_array = img_array.get('composite', img_array.get('image'))

    img = Image.fromarray(img_array.astype(np.uint8))

    # Convert grayscale
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert('L')
    elif img.mode == 'RGB':
        img = img.convert('L')

    # Invert nếu nền trắng
    if np.mean(np.array(img)) > 128:
        img = ImageOps.invert(img)

    image_preview = img.resize((28, 28), Image.LANCZOS)

    return image_preview
def preprocess_image(img):
    """
    img: image_preview
    """
    # Normalize (giống training CNN của bạn)
    img = np.array(img).astype("float32") / 255.0

    # reshape về (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    return img

# Hàm dự đoán
def predict_character(img_input, model_choice='CNN'):
    image_preview = input_image(img_input)
    img = preprocess_image(image_preview)

    if img is None:
        return 'Vui lòng vẽ hoặc tải ảnh!', None, ''

    # Nếu bạn có nhiều model thì map ở đây
    model_map = {
        'CNN': model  # model bạn đã train
    }

    model_selected = model_map.get(model_choice, model)

    # Predict (Keras -> trả về probability)
    logits = model_selected.predict(img, verbose=0)[0]

# Chuyển logits thành xác suất
    probs = tf.nn.softmax(logits).numpy()

    top5_idx = probs.argsort()[::-1][:5]
    top5_label = [str(i) for i in top5_idx]
    top5_prob = [float(probs[i]) for i in top5_idx]

    best_label = top5_label[0]
    best_conf = top5_prob[0]

    # ===== Vẽ bar chart =====
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(top5_label[::-1], top5_prob[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence')
    ax.set_title('Top-5 Predictions')

    for bar, prob in zip(bars, top5_prob[::-1]):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{prob:.2%}', va='center')

    plt.tight_layout()

    top5_text = '\n'.join(
        [f'{i+1}. {lbl}: {p:.2%}' for i, (lbl, p) in enumerate(zip(top5_label, top5_prob))]
    )

    result_text = f'🎯 Dự đoán: **{best_label}** ({best_conf:.2%})'

    return result_text, fig, top5_text


# Giao diện người dùng
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# ✍️ Nhận dạng chữ viết tay (CNN - Keras)")
    
    with gr.Row():
        with gr.Column(scale = 10):
            canvas = gr.Sketchpad(type='numpy')
            result = gr.Markdown()
            image_preview = gr.Image(label= "Image Preview")
            
        with gr.Column(scale = 9):
            plot = gr.Plot()
            
        with gr.Column(scale = 1):
            with gr.Accordion("Xác suất dự đoán:", open = False):
                text = gr.Textbox(lines = 5)

    btn = gr.Button("Dự đoán🔍")
    show = gr.Button("Show🔑")

    btn.click(
        fn=predict_character,
        inputs=canvas,
        outputs=[result, plot, text]
    )
    show.click(
        fn=input_image,
        inputs=canvas,
        outputs=image_preview
    )

demo.launch(debug = True)
model.save("mnist_model.h5")
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))