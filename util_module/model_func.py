import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import tensorflow as tf

from util_module import util_func

CLASSES = [
    'Zero padding',
    'Pon-Poff',
    'Poff-QRSon',
    'QRSon-Rpeak',
    'Rpeak-QRSoff',
    'QRSoff-Ton',
    'Ton-Toff',
    'Toff-Pon2'
]
COLORS = { # Zero padding given no color
    1: 'red',
    2: 'darkorange',
    3: 'yellow',
    4: 'green',
    5: 'blue',
    6: 'darkcyan',
    7: 'purple'
}

# Fixed for 80% train, 10% val, 10% test
def train_val_test_split(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=.1, shuffle=False, random_state=2023)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=len(X_test)/len(X_temp), shuffle=False, random_state=2023)

    return X_train, X_val, X_test, y_train, y_val, y_test

def generate_model(input_shape, output, lr=1e-5, n_layer=1):
    model = tf.keras.models.Sequential()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # Conv layers
    model.add(tf.keras.layers.Conv1D(8, 3, input_shape=input_shape, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'))

    # Bidirectional LSTM
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(781, input_shape=input_shape, return_sequences=True)))

    for i in range(n_layer-1):
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(781, return_sequences=True)))

    # Fully connected layers
    model.add(tf.keras.layers.Dense(output, activation='softmax'))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Plotting training and validation accuracy
def plot_acc_loss(model_h, save_to):
    util_func.make_dir(save_to)

    train_acc = model_h.history['accuracy']
    train_loss = model_h.history['loss']

    val_acc = model_h.history['val_accuracy']
    val_loss = model_h.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plotting training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_to}/Train_val_acc_loss.png', bbox_inches='tight')
    plt.show()

# classes is a list that contains class names in string
def calc_metrics(y_true, y_pred, save_to):
    util_func.make_dir(save_to)

    y_true = y_true.reshape(y_true.shape[0] * y_true.shape[1], y_true.shape[2])
    y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2])

    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'Recall': [],
        'Precision': [],
        'Specificity': [],
        'F1-score': [],
        'Accuracy': [],
        'Error rate': []
    }
    
    sum_tp = 0
    sum_fn = 0
    sum_fp = 0
    sum_tn = 0
    
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sum_tp += tp
        sum_fn += fn
        sum_fp += fp
        sum_tn += tn

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        specificity = tn / (tn + fp)
        f1 = 2 * (recall * precision) / (recall + precision)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        error = (fp + fn) / (tp + tn + fp + fn)
        
        metrics['Recall'].append(recall)
        metrics['Precision'].append(precision)
        metrics['Specificity'].append(specificity)
        metrics['F1-score'].append(f1)
        metrics['Accuracy'].append(accuracy)
        metrics['Error rate'].append(error)
    
    # Macro-averaged
    metrics['Recall'].append( np.sum(metrics['Recall']) / len(CLASSES) )
    metrics['Precision'].append( np.sum(metrics['Precision']) / len(CLASSES) )
    metrics['Specificity'].append( np.sum(metrics['Specificity']) / len(CLASSES) )
    metrics['F1-score'].append( np.sum(metrics['F1-score']) / len(CLASSES) )
    metrics['Accuracy'].append( np.sum(metrics['Accuracy']) / len(CLASSES) )
    metrics['Error rate'].append( np.sum(metrics['Error rate']) / len(CLASSES) )

    # Micro-averaged
    micro_recall = sum_tp / (sum_tp + sum_fn)
    micro_precision = sum_tp / (sum_tp + sum_fp)
    metrics['Recall'].append(micro_recall)
    metrics['Precision'].append(micro_precision)
    metrics['Specificity'].append(sum_tn / (sum_tn + sum_fp))
    metrics['F1-score'].append(2 * (micro_recall * micro_precision) / (micro_recall + micro_precision))
    metrics['Accuracy'].append( (sum_tp + sum_tn) / (sum_tp + sum_tn + sum_fp + sum_fn) )
    metrics['Error rate'].append( (sum_fp + sum_fn) / (sum_tp + sum_tn + sum_fp + sum_fn) )

    metrics_indexes = CLASSES + ['Macro-averaged', 'Micro-averaged']

    metrics_dataframe = pd.DataFrame(metrics, index=metrics_indexes)
    metrics_dataframe.to_csv(f'{save_to}/metrics.csv')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.savefig(f'{save_to}/CM.png', bbox_inches='tight')

def roc_pr(y_true, y_pred, save_to):
    util_func.make_dir(save_to)

    y_true = y_true.reshape(y_true.shape[0] * y_true.shape[1], y_true.shape[2])
    y_pred = y_pred.round().reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2])
    
    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, average_precision = {}, {}, {}

    # exclude zero padding on first index
    for i in range(1, 8):
        yt_i = y_true[:, i]
        yp_i = y_pred[:, i]
        fpr[i], tpr[i], _ = roc_curve(yt_i, yp_i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        precision[i], recall[i], _ = precision_recall_curve(yt_i, yp_i)
        average_precision[i] = average_precision_score(yt_i, yp_i)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for i in range(1, 8):
        plt.plot(
            fpr[i],
            tpr[i],
            color=COLORS[i],
            lw=2,
            label=f'Class {CLASSES[i]} (AUC = {roc_auc[i]:.2f})'
        )

    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.savefig(f'{save_to}/ROC_curve.jpg', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for i in range(1, 8):
        plt.plot(
            recall[i],
            precision[i],
            color=COLORS[i],
            lw=2,
            label=f'Class {CLASSES[i]} (AP = {average_precision[i]:.2f})',
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    fig.savefig(f'{save_to}/PR_curve.jpg', bbox_inches='tight')