import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import wfdb
from wfdb.processing import normalize_bound

from util_module import util_func

CURR_DIR = os.getcwd()
DATA_DIR = os.path.join(CURR_DIR, '../data/ludb')

# for better readability
SEGMENTS_STR = {
    0: 'Zero padding',
    1: 'Pon-Poff',
    2: 'Poff-QRSon',
    3: 'QRSon-Rpeak',
    4: 'Rpeak-QRSoff',
    5: 'QRSoff-Ton',
    6: 'Ton-Toff',
    7: 'Toff-Pon2'
}
SEGMENTS_NUM = {
    'Zero padding': 0,
    'Pon-Poff': 1,
    'Poff-QRSon': 2,
    'QRSon-Rpeak': 3,
    'Rpeak-QRSoff': 4,
    'QRSoff-Ton': 5,
    'Ton-Toff': 6,
    'Toff-Pon2': 7
}
SEGMENT_TO_COLOR = {
    1: 'red',
    2: 'darkorange',
    3: 'yellow',
    4: 'green',
    5: 'blue',
    6: 'darkcyan',
    7: 'purple'
}
LEADS = ['atr_avf', 'atr_avl', 'atr_avr', 'atr_i', 'atr_ii', 'atr_iii', 'atr_v1', 'atr_v2', 'atr_v3', 'atr_v4', 'atr_v5', 'atr_v6']

WAVELET_FUNCTION = 'coif5'
DECOMPOSITION_LEVEL = 8

class ECGSignal:
    def __init__(self, signal, samples, symbols):
        self.signal = util_func.denoise_dwt(signal, wavelet=WAVELET_FUNCTION, level=DECOMPOSITION_LEVEL)
        self.signal = normalize_bound(self.signal)
        self.samples = samples
        self.symbols = symbols

        self.segment_map, self.segment_start_end = self.segmentate()

    @staticmethod
    def load_ecg_signal(record_number, lead):
        record_dir = os.path.join(DATA_DIR, str(record_number))

        record = wfdb.rdrecord(record_dir)
        ann = wfdb.rdann(record_dir, lead)

        signal = record.p_signal[:, LEADS.index(lead)]
        samples = ann.sample
        symbols = ann.symbol

        return ECGSignal(signal, samples, symbols)

    @staticmethod
    def plot_signal_segments(signal, segment_map, ax=None, save_path=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(28, 3))
        # TODO: NO IDEA HOW I CAN SAVE THIS PASSED AX INSTANCE. So if ax is not None and save_path is not None, thing will surely break

        ax.plot(signal)

        segment_start_end = []
        segments = []
        legend_patches = []

        ptr_start = 0
        ptr_end = 0

        for i in range(1, len(segment_map)):
            curr_seg = segment_map[i]
            prev_seg = segment_map[i-1]
            if curr_seg != prev_seg:
                ptr_end = i - 1
                if(segment_map[ptr_start] not in [-1, 0]): # not annotated (-1) and zero pad (0)
                    segment_start_end.append((segment_map[ptr_start], (ptr_start, ptr_end)))
                    if(segment_map[ptr_start] not in segments):
                        segments.append(segment_map[ptr_start])
                ptr_start = i
        
        # add last segment
        ptr_end = len(segment_map) - 1
        if(segment_map[ptr_start] not in [-1, 0]):
            segment_start_end.append((segment_map[ptr_start], (ptr_start, ptr_end)))
            if(segment_map[ptr_start] not in segments):
                segments.append(segment_map[ptr_start])

        for seg, points in segment_start_end:
            start, end = points
            color = SEGMENT_TO_COLOR[seg]
            ax.axvspan(start, end, color=color, alpha=0.4)
        
        for seg in sorted(segments):
            patch = patches.Patch(color=SEGMENT_TO_COLOR[seg], label=SEGMENTS_STR[seg])
            legend_patches.append(patch)

        ax.legend(handles=legend_patches, loc='upper right')
        if save_path is not None:
            fig.savefig(save_path)
    
    '''
    -CURRENTLY NOT NEEDED
    -1d to 2d only
    -valid_values is a set of unique valid values
    -all values in segment_map that doesn't exist in valid_values will be ignored
    '''
    # @staticmethod
    # def one_hot_segment_map(segment_map, valid_values):
    #     num_classes = len(valid_values)

    #     res = np.zeros((segment_map.size, num_classes))

    #     for value in valid_values:
    #         res[np.where(segment_map == value), value] = 1
        
    #     return res

    def plot_signal_samples(self):
        _, ax = plt.subplots(figsize=(28, 3))

        ax.plot(self.signal)
        ax.scatter(self.samples, self.signal[self.samples], c='red')
    
    def plot_segments(self):
        _, ax = plt.subplots(figsize=(28, 3))

        ax.plot(self.signal)

        segments = []
        legend_patches = []
        
        for _, points in self.segment_start_end:
            start, end = points
            color = SEGMENT_TO_COLOR[self.segment_map[start]]
            ax.axvspan(start, end, color=color, alpha=0.4)

            if self.segment_map[start] not in segments:
                segments.append(self.segment_map[start])
        
        # make patches for existing segments
        for seg in sorted(segments):
            patch = patches.Patch(color=SEGMENT_TO_COLOR[seg], label=SEGMENTS_STR[seg])
            legend_patches.append(patch)

        ax.legend(handles=legend_patches, loc='upper right')
        
    def segmentate(self):
        segment_map = np.full(len(self.signal), -1)
        segment_start_end = []

        prev_symbol = None
        for start, peak, end in util_func.grouped(self.samples):
            symbol_idx = np.where(self.samples == peak)[0][0]

            curr_symbol = self.symbols[symbol_idx]

            if curr_symbol == 'p':
                if prev_symbol is not None and prev_symbol == 't':
                    segment_map[prev_end:start] = SEGMENTS_NUM['Toff-Pon2']
                    segment_start_end.append((SEGMENTS_NUM['Toff-Pon2'], (prev_end, start)))

                segment_map[start:end] = SEGMENTS_NUM['Pon-Poff']
                segment_start_end.append((SEGMENTS_NUM['Pon-Poff'], (start, end)))
                

            elif curr_symbol == 'N':
                if prev_symbol == 'p':
                    segment_map[prev_end:start] = SEGMENTS_NUM['Poff-QRSon']
                    segment_start_end.append((SEGMENTS_NUM['Poff-QRSon'], (prev_end, start)))

                segment_map[start:peak] = SEGMENTS_NUM['QRSon-Rpeak']
                segment_start_end.append((SEGMENTS_NUM['QRSon-Rpeak'], (start, peak)))

                segment_map[peak:end] = SEGMENTS_NUM['Rpeak-QRSoff']
                segment_start_end.append((SEGMENTS_NUM['Rpeak-QRSoff'], (peak, end)))

            elif curr_symbol == 't':
                if prev_symbol == 'N':
                    segment_map[prev_end:start] = SEGMENTS_NUM['QRSoff-Ton']
                    segment_start_end.append((SEGMENTS_NUM['QRSoff-Ton'], (prev_end, start)))

                segment_map[start:end] = SEGMENTS_NUM['Ton-Toff']
                segment_start_end.append((SEGMENTS_NUM['Ton-Toff'], (start, end)))
            
            prev_symbol = curr_symbol
            prev_end = end
        
        return segment_map, segment_start_end


    # returns signal and segment_map
    def cut_per_beat(self):
        normal_beat_seq = [1, 2, 3, 4, 5, 6, 7] # normal beat segments sequence
        seq_to_compare = []
        p_start_ptr = 0
        t_end_ptr = 0
        prev_end = None
        beats = [] # beats list contains tuple (signal, segment_map) of each beat

        for seg, points in self.segment_start_end:
            start, end = points

            # if start - prev_end is not 0 that means theres a gap between the segment (not a normal beat)
            if(prev_end is None) or (start - prev_end == 0):
                seq_to_compare.append(seg)

            prev_end = end

            if(seg == SEGMENTS_NUM['Pon-Poff']):
                p_start_ptr = start

                seq_to_compare = [1]
            
            elif(seg == SEGMENTS_NUM['Toff-Pon2']):
                t_end_ptr = end

                if(seq_to_compare == normal_beat_seq):
                    signal = self.signal[p_start_ptr:t_end_ptr]
                    segment_map = self.segment_map[p_start_ptr:t_end_ptr] 

                    beats.append((signal, segment_map))
        
        return beats

    def cut_p_to_p(self):
        prev_p_start = None
        beats = []
        for seg, points in self.segment_start_end:
            start, _ = points

            if (seg == SEGMENTS_NUM['Pon-Poff']) and (prev_p_start is None):
                prev_p_start = start
                break
            
            elif (seg == SEGMENTS_NUM['Pon-Poff']):
                signal = self.signal[prev_p_start:start]
                segment_map = self.segment_map[prev_p_start:start]

                beats.append((signal, segment_map))
        
        return beats


            