import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from util_module import util_func
from util_module.ecg_signal import ECGSignal

CLASSES = [
    'Pon-Poff',
    'Poff-QRSon',
    'QRSon-Rpeak',
    'Rpeak-QRSoff',
    'QRSoff-Ton',
    'Ton-Toff',
    'Toff-Pon2',
    'Zero padding',
]
COLORS = { # Zero padding given no color
    0: 'red',
    1: 'darkorange',
    2: 'yellow',
    3: 'green',
    4: 'blue',
    5: 'darkcyan',
    6: 'purple'
}

#DEPRECATED
def generate_model(input_shape, output, lr=1e-5, n_layer=1):
    model = tf.keras.models.Sequential()
    opt = tf.keras.optimizers.RMSprop(learning_rate=lr)

    # Conv layers
    model.add(tf.keras.layers.Conv1D(8, 3, input_shape=input_shape, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'))

    # Bidirectional LSTM
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_shape[0], input_shape=input_shape, return_sequences=True)))

    for i in range(n_layer-1):
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_shape[0], return_sequences=True)))

    # Fully connected layers
    model.add(tf.keras.layers.Dense(output, activation='softmax'))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Plotting training and validation accuracy
def plot_acc_loss(model_h, save_to):
    util_func.make_dir(save_to)

    train_acc = model_h['accuracy']
    train_loss = model_h['loss']

    val_acc = model_h['val_accuracy']
    val_loss = model_h['val_loss']

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
    metrics_dataframe = metrics_dataframe.applymap(lambda x: np.round(x * 100, 2))
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

    # exclude zero padding
    for i in range(0, 7):
        yt_i = y_true[:, i]
        yp_i = y_pred[:, i]
        fpr[i], tpr[i], _ = roc_curve(yt_i, yp_i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        precision[i], recall[i], _ = precision_recall_curve(yt_i, yp_i)
        average_precision[i] = average_precision_score(yt_i, yp_i)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for i in range(0, 7):
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
    for i in range(0, 7):
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

#DEPRECATED
# def generate_results(train_set, val_set, test_set, zpad_length, lead):
#     # plot_segments wrapper strictly for this function
#     def plot_segments(X, y, zpad, idx, save_path):
#         signal = X[idx].flatten()
#         segment_map = y[idx].argmax(axis=1)

#         beat_span = len(signal) - zpad[idx]

#         signal = signal[:beat_span]
#         segment_map = segment_map[:beat_span]

#         ECGSignal.plot_signal_segments(signal, segment_map, save_path)
    
#     start_time = time.time()

#     X_train, y_train = train_set
#     X_val, y_val = val_set
#     X_test, y_test = test_set

#     zpad_length_train, zpad_length_val, zpad_length_test = zpad_length

#     # Good fit model architecture
#     lr = 0
#     n_layer = 0
#     bs = 0
#     epoch = 0

#     model_name = f'ConvBiLSTM--{lead}--LR_{lr}-Nlayer_{n_layer}-BS_{bs}-Epoch_{epoch}'
#     result_path = f'../result/{model_name}'
#     save_to_train = f'{result_path}/train'
#     save_to_val = f'{result_path}/val'
#     save_to_test = f'{result_path}/test'

#     util_func.make_dir(f'{save_to_train}/delineation')
#     util_func.make_dir(f'{save_to_val}/delineation')
#     util_func.make_dir(f'{save_to_test}/delineation')

#     model = generate_model((X_train.shape[1], 1), 8, lr=lr, n_layer=n_layer)
#     history = model.fit(X_train, y_train, epochs=epoch, batch_size=bs, validation_data=(X_val, y_val))

#     model.save(f'{model_name}.h5')
#     plot_acc_loss(history, result_path)

#     # PREDICT TRAIN
#     y_pred_train = model.predict(X_train)
#     calc_metrics(y_train, y_pred_train, save_to_train)
#     roc_pr(y_train, y_pred_train, save_to_train)
    
#     for i in range(0, 41, 10):
#         plot_segments(X_train, y_train, zpad=zpad_length_train, idx=i, save_path=f'{save_to_train}/delineation/Expert_annotated_{i}.jpg') # Expert annotated
#         plot_segments(X_train, y_pred_train, zpad=zpad_length_train, idx=i, save_path=f'{save_to_train}/delineation/Prediction_{i}.jpg') # Prediction
#     # ====================================================

#     # PREDICT VALIDATION
#     y_pred_val = model.predict(X_val)
#     calc_metrics(y_val, y_pred_val, save_to_val)
#     roc_pr(y_val, y_pred_val, save_to_val)

#     for i in range(0, 41, 10):
#         plot_segments(X_val, y_val, zpad=zpad_length_val, idx=i, save_path=f'{save_to_val}/delineation/Expert_annotated_{i}.jpg') # Expert annotated
#         plot_segments(X_val, y_pred_val, zpad=zpad_length_val, idx=i, save_path=f'{save_to_val}/delineation/Prediction_{i}.jpg') # Prediction
#     # ====================================================

#     # PREDICT test
#     y_pred_test = model.predict(X_test)
#     calc_metrics(y_test, y_pred_test, save_to_test)
#     roc_pr(y_test, y_pred_test, save_to_test)

#     for i in range(0, 41, 10):
#         plot_segments(X_test, y_test, zpad=zpad_length_test, idx=i, save_path=f'{save_to_test}/delineation/Expert_annotated_{i}.jpg') # Expert annotated
#         plot_segments(X_test, y_pred_test, zpad=zpad_length_test, idx=i, save_path=f'{save_to_test}/delineation/Prediction_{i}.jpg') # Prediction
#     # ====================================================

#     end_time = time.time()
#     time_elapsed = end_time - start_time
#     with open(f'{result_path}/Model_info.txt', 'w') as info_file:
#         info_file.write(f'{model_name}\n')
#         info_file.write(f'Learning Rate: {lr} | n_layer: {n_layer} | Batch Size: {bs}\n | Epoch: {epoch}')
#         info_file.write(f'Time elapsed: {time_elapsed:.2f} seconds\n')

def generate_results(model, model_history, model_info, train_set, val_set, test_set):
    '''
    model: trained model
    model_history: dictionary of model training history
    model_info: dictionary, must atleast have the "name" key
    '''

    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set
    
    result_path = f'../result/{model_info["name"]}'
    save_to_train = f'{result_path}/train'
    save_to_val = f'{result_path}/val'
    save_to_test = f'{result_path}/test'

    util_func.make_dir(f'{save_to_train}/delineation')
    util_func.make_dir(f'{save_to_val}/delineation')
    util_func.make_dir(f'{save_to_test}/delineation')

    model.save(f'{result_path}/{model_info["name"]}.h5')

    plot_acc_loss(model_history, result_path)

    # PREDICT TRAIN
    y_pred_train = model.predict(X_train)
    calc_metrics(y_train, y_pred_train, save_to_train)
    roc_pr(y_train, y_pred_train, save_to_train)
    
    for i in range(0, 41, 10):
        util_func.plot_rhytm_gt_pred(X_train,
                                     y_train,
                                     y_pred_train,
                                     None,
                                     i,
                                     fig_title=f'Lead {model_info["lead"]} - Train Set',
                                     length=1,
                                     save_path=f'{save_to_train}/delineation/Beat_{i}.jpg')
    # ====================================================

    # PREDICT VALIDATION
    y_pred_val = model.predict(X_val)
    calc_metrics(y_val, y_pred_val, save_to_val)
    roc_pr(y_val, y_pred_val, save_to_val)

    for i in range(0, 41, 10):
        util_func.plot_rhytm_gt_pred(X_val,
                                     y_val,
                                     y_pred_val,
                                     None,
                                     i,
                                     fig_title=f'Lead {model_info["lead"]} - Validation Set',
                                     length=1,
                                     save_path=f'{save_to_val}/delineation/Beat_{i}.jpg')
    # ====================================================

    # PREDICT test
    y_pred_test = model.predict(X_test)
    calc_metrics(y_test, y_pred_test, save_to_test)
    roc_pr(y_test, y_pred_test, save_to_test)

    for i in range(0, 41, 10):
        util_func.plot_rhytm_gt_pred(X_test,
                                     y_test,
                                     y_pred_test,
                                     None,
                                     i,
                                     fig_title=f'Lead {model_info["lead"]} - Test Set',
                                     length=1,
                                     save_path=f'{save_to_test}/delineation/Beat_{i}.jpg')
    # ====================================================

    with open(f'{result_path}/Model_info.txt', 'w') as info_file:
        model.summary(print_fn=lambda x: info_file.write(x + '\n'))

        for key in model_info.keys():
            info_file.write(f"{key}: {model_info[key]}\n")
        