"""
CS 5720 Neural Network and Deep Learning, Fall 2026
Home Assignment 2, all questions in one script.

Student: Jeffrey Agnitsch
Student ID: 7005146960

Run everything:
    python home_assignment_2.py
Run a subset:
    python home_assignment_2.py --q 3,4,5
Q1/Q2 training knobs (defaults are the assignment-scale settings):
    python home_assignment_2.py --q 1 --epochs 10 --rnn_units 1024
    python home_assignment_2.py --q 1 --text little_prince.txt --seed "The little prince"
    python home_assignment_2.py --q 2 --epochs 3
    python home_assignment_2.py --q 4 --image photo.jpg

Sections
    Q1  Character-level LSTM text generation with temperature sampling
    Q2  LSTM sentiment classifier on IMDB, confusion matrix + classification report
    Q3  2-D convolution of a 5x5 input with a 3x3 kernel, stride 1/2, VALID/SAME
    Q4  Sobel edge detection (OpenCV) and 2x2 max / average pooling (Keras)
    Q5  Simplified AlexNet and a small ResNet built from residual blocks
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# One seed for everything so the random 4x4 in Q4 and the text samples in Q1
# are reproducible run to run.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Wide printing so 5x5 and 4x4 matrices do not wrap.
np.set_printoptions(linewidth=140, precision=3, suppress=True)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =========================================================================== #
# Q1: Character-level text generation with an LSTM
# =========================================================================== #
SHAKESPEARE_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)


def q1_load_text(path):
    """Return the corpus as one string. Default corpus is the Shakespeare file
    used in the course deck and the TensorFlow text-generation tutorial."""
    if path is None:
        # Download Shakespeare corpus from TensorFlow's public storage
        path = keras.utils.get_file("shakespeare.txt", SHAKESPEARE_URL)
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")


def q1_build_vocab(text):
    """Character <-> integer lookup. Index into `chars` is the character id."""
    chars = sorted(set(text))                     # all unique characters in the corpus
    char2idx = {c: i for i, c in enumerate(chars)}  # char -> integer mapping
    idx2char = np.array(chars)                     # integer -> char mapping (reverse)
    return chars, char2idx, idx2char


def q1_make_dataset(text_as_int, seq_length, batch_size, buffer_size=10_000):
    """Cut the integer stream into (input, target) windows.

    Every example is seq_length + 1 characters. Input = chars [0, L),
    target = chars [1, L], i.e. the same window shifted right by one, so the
    model predicts the next character at every position. Fixed-length windows
    are truncated backpropagation through time (the deck's seq_length idea).
    """
    char_ds = tf.data.Dataset.from_tensor_slices(text_as_int)
    windows = char_ds.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]

    return (
        windows.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)  # fixed batch size for the stateful twin
        .prefetch(tf.data.AUTOTUNE)
    )


def q1_build_model(vocab_size, embedding_dim, rnn_units, batch_size, stateful):
    """Embedding -> LSTM -> Dense(logits).

    Embedding : a one-hot input times W_xh only selects one column of W_xh, so a
                learned embedding table replaces the one-hot input outright.
    LSTM      : return_sequences=True gives a hidden state, hence a prediction,
                at every time step. stateful=True (generation only) keeps the
                (h, c) state across calls so we can feed one character at a time.
    Dense     : hidden state -> unnormalised scores over the vocabulary. No
                softmax here: the loss applies it, and the sampler divides by
                temperature first.
    """
    inputs = keras.Input(shape=(None,), batch_size=batch_size, dtype="int32")
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = layers.LSTM(rnn_units, return_sequences=True, stateful=stateful,
                    recurrent_initializer="glorot_uniform")(x)
    logits = layers.Dense(vocab_size)(x)
    return keras.Model(inputs, logits, name="char_lstm")


def q1_reset_rnn_state(model):
    """Zero the LSTM's carried (h, c). Keras 3 renamed reset_states() to
    reset_state(); support both so the script runs on TF 2.15 and 2.16+."""
    for layer in model.layers:
        if isinstance(layer, layers.LSTM):
            if hasattr(layer, "reset_state"):
                layer.reset_state()
            else:
                layer.reset_states()


def q1_generate_text(model, seed, char2idx, idx2char, num_generate=500, temperature=1.0):
    """Sample num_generate characters one at a time.

    Temperature scaling:  p = softmax(z / T)
        T -> 0 : largest logit dominates, sampling becomes argmax (deterministic,
                 repetitive).
        T = 1  : sample from the distribution the model learned.
        T > 1  : logits squeezed together, distribution flattens toward uniform,
                 output gets more random and more error-prone.
    """
    # Convert seed string to integer ids and wrap in a batch dimension
    input_ids = tf.expand_dims([char2idx[c] for c in seed], 0)  # (1, len(seed))
    generated = []
    q1_reset_rnn_state(model)  # clear any leftover LSTM state from prior calls
    for _ in range(num_generate):
        logits = model(input_ids)[:, -1, :]      # take logits at the last time step -> (1, vocab)
        logits = logits / temperature            # temperature scaling: divide logits before softmax
        # tf.random.categorical draws one sample from the scaled probability distribution
        next_id = tf.random.categorical(logits, num_samples=1)[-1, 0].numpy()
        generated.append(idx2char[next_id])      # map integer id back to character
        input_ids = tf.expand_dims([next_id], 0)  # feed the sampled character back as next input
    return seed + "".join(generated)


def run_q1(args):
    banner("Q1: Character-level LSTM text generation")
    # Step 1: Load and tokenize the text corpus into character-level integer ids
    text = q1_load_text(args.text)
    chars, char2idx, idx2char = q1_build_vocab(text)
    vocab_size = len(chars)
    text_as_int = np.array([char2idx[c] for c in text], dtype=np.int32)
    print(f"Corpus: {len(text):,} characters, {vocab_size} unique")

    # Step 2: Create training dataset of (input, target) character windows
    dataset = q1_make_dataset(text_as_int, args.seq_length, args.batch_size)

    # Step 3: Build the LSTM model (non-stateful for training with batches)
    model = q1_build_model(vocab_size, args.embedding_dim, args.rnn_units,
                           batch_size=args.batch_size, stateful=False)
    model.summary()
    # SparseCategoricalCrossentropy with from_logits=True applies softmax internally
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Step 4: Train the model and save weights after each epoch
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "q1_char_lstm.weights.h5")
    t0 = time.time()
    hist = model.fit(dataset, epochs=args.epochs,
                     callbacks=[keras.callbacks.ModelCheckpoint(ckpt, save_weights_only=True)])
    print(f"Training time: {time.time() - t0:.1f}s")
    print("Loss per epoch:", [round(v, 4) for v in hist.history["loss"]])

    # Step 5: Rebuild the model with batch_size=1 and stateful=True for generation.
    # Stateful mode keeps the LSTM hidden state between calls so we can feed
    # one character at a time and have the context accumulate.
    gen_model = q1_build_model(vocab_size, args.embedding_dim, args.rnn_units,
                               batch_size=1, stateful=True)
    gen_model.load_weights(ckpt)

    # Generate text at four different temperatures to show the effect on randomness
    for T in (0.2, 0.5, 1.0, 1.5):
        print(f"\n----- temperature = {T} -----")
        print(q1_generate_text(gen_model, args.seed, char2idx, idx2char,
                               num_generate=args.num_generate, temperature=T))


# =========================================================================== #
# Q2: IMDB sentiment classification with an LSTM
# =========================================================================== #
def q2_build_model(vocab_size, maxlen, embedding_dim=128, lstm_units=64):
    """Embedding -> LSTM -> Dense(1, sigmoid).

    Many-to-one RNN: the LSTM reads the whole padded review and only its final
    hidden state (return_sequences=False) feeds the classifier. mask_zero=True
    tells the LSTM to skip the padding token (id 0) so padded positions do not
    corrupt the final state.
    """
    model = keras.Sequential([
        keras.Input(shape=(maxlen,)),
        layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.0),
        layers.Dense(1, activation="sigmoid"),
    ], name="imdb_lstm")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_q2(args):
    from sklearn.metrics import classification_report, confusion_matrix

    banner("Q2: IMDB sentiment classification with an LSTM")
    vocab_size, maxlen = 10_000, 200  # keep top 10k words, pad/truncate to 200 tokens

    # The Keras IMDB dataset is already tokenized: each review is a list of
    # word ids, most frequent word = 1. num_words keeps only the top-N words
    # (rarer words map to the OOV id 2). That is the "tokenization" step.
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
    print(f"Train reviews: {len(x_train):,}   Test reviews: {len(x_test):,}")
    lens = [len(r) for r in x_train]
    print(f"Review length: mean {np.mean(lens):.0f}, median {np.median(lens):.0f}, max {max(lens)}")

    # Padding: LSTM batches need equal-length sequences. Pre-padding/truncation
    # keeps the END of each review, which is where the verdict usually lives.
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen,
                                                         padding="pre", truncating="pre")
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen,
                                                        padding="pre", truncating="pre")
    print(f"Padded shape: {x_train.shape}")

    # Build and train the LSTM sentiment classifier
    model = q2_build_model(vocab_size, maxlen)
    model.summary()
    hist = model.fit(x_train, y_train, epochs=args.epochs, batch_size=128,
                     validation_split=0.2, verbose=1)

    # Get predicted probabilities on the test set
    probs = model.predict(x_test, batch_size=512, verbose=0).ravel()
    # Threshold at 0.5: reviews with P(positive) >= 0.5 are classified positive
    y_pred = (probs >= 0.5).astype(int)

    # Generate confusion matrix: shows TN, FP, FN, TP counts
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows = true, cols = predicted; 0 = negative, 1 = positive)")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Classification report: precision, recall, F1-score per class
    print("\nClassification report")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"],
                                digits=4, zero_division=0))

    # Sweep the decision threshold to demonstrate the precision-recall tradeoff.
    # Lower threshold -> higher recall (catch more positives) but lower precision.
    # Higher threshold -> higher precision (fewer false positives) but lower recall.
    print("Threshold sweep on the positive class (precision vs recall):")
    print(f"{'thr':>5} {'precision':>10} {'recall':>8}")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        yp = (probs >= thr).astype(int)
        tp_ = np.sum((yp == 1) & (y_test == 1))   # true positives
        fp_ = np.sum((yp == 1) & (y_test == 0))   # false positives
        fn_ = np.sum((yp == 0) & (y_test == 1))   # false negatives
        prec = tp_ / max(tp_ + fp_, 1)            # precision = TP / (TP + FP)
        rec = tp_ / max(tp_ + fn_, 1)             # recall    = TP / (TP + FN)
        print(f"{thr:>5.1f} {prec:>10.4f} {rec:>8.4f}")


# =========================================================================== #
# Q3: Convolution with different stride and padding
# =========================================================================== #
Q3_INPUT = np.array([
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9,  10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
], dtype=np.float32)

Q3_KERNEL = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0],
], dtype=np.float32)


def q3_conv2d_numpy(x, k, stride, padding):
    """Plain-NumPy 2-D cross-correlation (what every DL framework calls
    "convolution": the kernel is NOT flipped). Reproduces TensorFlow's
    padding rules exactly.

    VALID : no padding. out = floor((n - k) / s) + 1
    SAME  : out = ceil(n / s); total pad = max((out - 1) * s + k - n, 0),
            split floor(pad/2) on top/left and the remainder on bottom/right
            (TensorFlow puts the extra row/column at the bottom/right).
    """
    n, kk = x.shape[0], k.shape[0]
    if padding == "VALID":
        out = (n - kk) // stride + 1
        xp = x
    elif padding == "SAME":
        out = -(-n // stride)                         # ceil(n / stride)
        pad_total = max((out - 1) * stride + kk - n, 0)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        xp = np.pad(x, ((pad_before, pad_after), (pad_before, pad_after)))
    else:
        raise ValueError(padding)

    y = np.zeros((out, out), dtype=np.float32)
    for i in range(out):
        for j in range(out):
            r, c = i * stride, j * stride
            y[i, j] = np.sum(xp[r:r + kk, c:c + kk] * k)   # elementwise mult + sum
    return y


def q3_conv2d_tf(x, k, stride, padding):
    """Same operation via tf.nn.conv2d. TF wants NHWC input and HWIO kernel."""
    x4 = tf.constant(x.reshape(1, *x.shape, 1))          # (batch=1, H, W, C=1)
    k4 = tf.constant(k.reshape(*k.shape, 1, 1))          # (kH, kW, in=1, out=1)
    y = tf.nn.conv2d(x4, k4, strides=stride, padding=padding)
    return y.numpy()[0, :, :, 0]


def run_q3(args):
    banner("Q3: Convolution with different stride and padding")
    print("Input 5x5:\n", Q3_INPUT)
    print("Kernel 3x3:\n", Q3_KERNEL)
    # Run all four stride/padding combinations and verify NumPy matches TensorFlow
    for stride, padding in [(1, "VALID"), (1, "SAME"), (2, "VALID"), (2, "SAME")]:
        y_np = q3_conv2d_numpy(Q3_INPUT, Q3_KERNEL, stride, padding)  # manual NumPy implementation
        y_tf = q3_conv2d_tf(Q3_INPUT, Q3_KERNEL, stride, padding)    # TensorFlow's built-in conv2d
        assert np.allclose(y_np, y_tf), "NumPy and TensorFlow disagree"  # sanity check
        print(f"\nStride = {stride}, Padding = '{padding}'  ->  output {y_tf.shape[0]}x{y_tf.shape[1]}")
        print(y_tf)


# =========================================================================== #
# Q4: Sobel edge detection and pooling
# =========================================================================== #
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)


def q4_load_gray_image(path):
    """Load the user's image as grayscale, or synthesise one with clean
    horizontal and vertical edges so the script runs with no external file."""
    import cv2
    if path is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img
    # Synthesise a test image with known edges if no image file is provided
    img = np.full((256, 256), 40, dtype=np.uint8)            # dark gray background
    cv2.rectangle(img, (40, 40), (140, 140), 200, -1)        # bright square: 4 edges
    cv2.circle(img, (180, 170), 50, 120, -1)                 # circle: edges at all angles
    cv2.line(img, (20, 230), (236, 230), 255, 3)             # horizontal line -> Sobel-Y responds
    cv2.line(img, (230, 20), (230, 200), 255, 3)             # vertical line   -> Sobel-X responds
    return img


def run_q4_task1(args):
    import cv2
    import matplotlib
    matplotlib.use("Agg")                    # render to file; works headless and in Colab
    import matplotlib.pyplot as plt

    banner("Q4 Task 1: Sobel edge detection with OpenCV")
    img = q4_load_gray_image(args.image)
    print(f"Image shape: {img.shape}, dtype {img.dtype}")

    # cv2.filter2D computes cross-correlation with the given kernel, exactly the
    # matrices from the assignment. ddepth=CV_64F keeps negative responses
    # (an edge going dark->bright vs bright->dark has opposite sign).
    # Apply Sobel-X (detects vertical edges) and Sobel-Y (detects horizontal edges)
    # using cv2.filter2D with the exact kernels from the assignment
    gx = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, SOBEL_X)
    gy = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, SOBEL_Y)
    # Convert to absolute value and scale to 0..255 for display
    gx_disp = cv2.convertScaleAbs(gx)
    gy_disp = cv2.convertScaleAbs(gy)
    # Verify our manual filter matches OpenCV's built-in Sobel function
    assert np.allclose(gx, cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3))
    assert np.allclose(gy, cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3))

    # Plot original image alongside both Sobel-filtered results side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, title in zip(axes, [img, gx_disp, gy_disp],
                             ["Original", "Sobel-X (vertical edges)", "Sobel-Y (horizontal edges)"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    # Save the figure to the outputs directory
    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, "q4_sobel.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"Saved figure: {out_png}")
    try:
        plt.show()                            # no-op under Agg; shows inline in notebooks
    except Exception:
        pass


def run_q4_task2(args):
    banner("Q4 Task 2: 2x2 max pooling and average pooling with Keras")
    mat = np.random.randint(0, 10, size=(4, 4)).astype(np.float32)
    # Keras pooling layers expect 4D input: (batch, height, width, channels)
    x = mat.reshape(1, 4, 4, 1)
    # Max pooling: take the largest value in each 2x2 window
    max_pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x).numpy()[0, :, :, 0]
    # Average pooling: compute the mean of each 2x2 window
    avg_pooled = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x).numpy()[0, :, :, 0]
    print("Original 4x4:\n", mat)
    print("\n2x2 max pooled (2x2):\n", max_pooled)
    print("\n2x2 average pooled (2x2):\n", avg_pooled)


def run_q4(args):
    run_q4_task1(args)
    run_q4_task2(args)


# =========================================================================== #
# Q5: AlexNet and a ResNet-like model
# =========================================================================== #
def q5_build_alexnet(input_shape=(227, 227, 3), num_classes=10):
    """Simplified AlexNet, layer list per the assignment.

    Padding is not specified in the assignment. The first 11x11/stride-4 conv
    is 'valid' (227 -> 55, as in the original paper); every later conv is
    'same' so spatial size only shrinks at the pooling layers, which is how
    Krizhevsky et al. (2012) built it: 55 -> 27 -> 13 -> 13 -> 13 -> 6.
    """
    # Build AlexNet using the Sequential API (layers stack linearly)
    model = keras.Sequential(name="AlexNet_simplified")
    model.add(keras.Input(shape=input_shape))
    # Convolutional feature extraction layers
    model.add(layers.Conv2D(96, (11, 11), strides=4, activation="relu", padding="valid"))   # 227->55
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))                             # 55->27
    model.add(layers.Conv2D(256, (5, 5), activation="relu", padding="same"))                # 27->27
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))                             # 27->13
    model.add(layers.Conv2D(384, (3, 3), activation="relu", padding="same"))                # 13->13
    model.add(layers.Conv2D(384, (3, 3), activation="relu", padding="same"))                # 13->13
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))                # 13->13
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))                             # 13->6
    # Fully connected classifier layers
    model.add(layers.Flatten())                                                             # 6*6*256=9216
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))                                                          # 50% dropout
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))                                                          # 50% dropout
    model.add(layers.Dense(num_classes, activation="softmax"))                               # output layer
    return model


def residual_block(input_tensor, filters=64):
    """Two 3x3 convs with an identity skip connection.

    conv1 -> ReLU -> conv2 (NO activation) -> Add(input) -> ReLU.
    Putting the add BEFORE the final activation is the He et al. (2016)
    ordering: the block learns a residual F(x), and the output is
    ReLU(F(x) + x). padding='same' keeps H and W equal so the add is legal;
    the channel count must already equal `filters` (the stem conv ensures it),
    otherwise a 1x1 projection conv would be needed on the skip path.
    """
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(input_tensor)  # first conv + ReLU
    x = layers.Conv2D(filters, (3, 3), padding="same", activation=None)(x)              # second conv, NO activation
    x = layers.Add()([x, input_tensor])            # skip connection: add input to learned residual F(x)
    return layers.Activation("relu")(x)            # ReLU after the addition: output = ReLU(F(x) + x)


def q5_build_resnet(input_shape=(224, 224, 3), num_classes=10):
    """Stem conv (7x7/2, 64 filters) -> two residual blocks -> Flatten ->
    Dense(128) -> softmax. Functional API is required because the skip
    connection makes the graph non-sequential."""
    # Functional API is required because skip connections make the graph non-sequential
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=2, padding="same", activation="relu")(inputs)  # stem conv
    x = residual_block(x, 64)   # first residual block with skip connection
    x = residual_block(x, 64)   # second residual block with skip connection
    x = layers.Flatten()(x)     # flatten spatial dimensions for the dense layers
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)  # 10-class output
    return keras.Model(inputs, outputs, name="ResNet_like")


def run_q5(args):
    banner("Q5 Task 1: Simplified AlexNet")
    q5_build_alexnet().summary()
    banner("Q5 Task 2: Residual block and ResNet-like model")
    q5_build_resnet().summary()


# =========================================================================== #
# Entry point
# =========================================================================== #
def main():
    """Parse command-line arguments and run the selected question(s)."""
    ap = argparse.ArgumentParser(description="CS 5720 Home Assignment 2")
    ap.add_argument("--q", default="1,2,3,4,5", help="comma-separated question numbers to run")
    ap.add_argument("--out_dir", default="./outputs", help="directory for output files")
    # Q1 arguments
    ap.add_argument("--text", default=None, help="Q1 corpus .txt (default: Shakespeare)")
    ap.add_argument("--seq_length", type=int, default=100, help="Q1 training sequence length")
    ap.add_argument("--batch_size", type=int, default=64, help="Q1 training batch size")
    ap.add_argument("--embedding_dim", type=int, default=256, help="Q1 embedding dimension")
    ap.add_argument("--rnn_units", type=int, default=1024, help="Q1 LSTM hidden units")
    ap.add_argument("--seed", default="ROMEO: ", help="Q1 seed string for text generation")
    ap.add_argument("--num_generate", type=int, default=400, help="Q1 characters to generate")
    # Q1 and Q2 shared argument
    ap.add_argument("--epochs", type=int, default=10, help="training epochs (Q1 and Q2)")
    # Q4 arguments
    ap.add_argument("--image", default=None, help="Q4 grayscale image path (default: synthetic)")
    args = ap.parse_args()

    # Map question numbers to their runner functions and execute in order
    runners = {"1": run_q1, "2": run_q2, "3": run_q3, "4": run_q4, "5": run_q5}
    for q in [s.strip() for s in args.q.split(",") if s.strip()]:
        runners[q](args)


if __name__ == "__main__":
    main()
