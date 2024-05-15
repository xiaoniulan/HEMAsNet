from scipy import io
import numpy as np
from scipy.signal import butter, filtfilt
import os
import random


def read_data():
    mat_data = io.loadmat('./data_8trial.mat')###load_dataset
    return mat_data


def create_folder(folder_num=None):
    for i in range(1, folder_num + 1):
        folder_name = f"{i}"
        os.makedirs(folder_name, exist_ok=True)


def band_extraction(eeg_data, frequency_range):
    sampling_rate = 250
    alpha_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):
        alpha_data[i, :] = butter_bandpass_filter(eeg_data[i, :], frequency_range[0], frequency_range[1], sampling_rate)
    return alpha_data


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def data_allocation(remaining_users=None, user_indices=None, control_subjects=None, subject_category_id=None,
                    folder_num=None):
    folder_index = 0
    for user_index in user_indices:
        folder_name = f"{folder_index + 1}"
        signal_num = control_subjects[0, user_index].shape[1]
        for signal_id in range(signal_num):
            eeg_data = control_subjects[0, user_index][0, signal_id]

            delta_band = (1, 4)
            theta_band = (4, 8)
            alpha_band = (8, 13)
            beta_band = (13, 30)
            gamma_band = (30, 50)

            delta_data = band_extraction(eeg_data, delta_band)
            theta_data = band_extraction(eeg_data, theta_band)
            alpha_data = band_extraction(eeg_data, alpha_band)
            beta_data = band_extraction(eeg_data, beta_band)
            gamma_data = band_extraction(eeg_data, gamma_band)

            combined_eeg_data = np.stack((delta_data, theta_data, alpha_data, beta_data, gamma_data), axis=1)
            selected_channels = [x-1 for x in [32,26,22,27,23,18,24,19,12,17,15,1,2,9,123,3,10,124,4,5,16,11]]

            segmented_eeg_data = [combined_eeg_data[selected_channels, :, i:i + 500] for i in range(0, 2500, 500)]####[channel,bands,points]这里改成22导联
            segment_num = len(segmented_eeg_data)

            for segment_id in range(segment_num):
                file_name = f'{subject_category_id}-{user_index + 1}-{signal_id + 1}-{segment_id + 1}.mat'
                all_file_name = os.path.join(folder_name, file_name)
                io.savemat(all_file_name, {'content': segmented_eeg_data[segment_id]})

        if remaining_users > 0:
            remaining_users -= 1
            folder_index += 1
        else:
            folder_index = (folder_index + 1) % folder_num


def deal_data():
    folder_num = 10
    create_folder(folder_num)
    category_str = ['control', 'depression']
    for i in range(2):
        if i == 0:
            subject_category_id = "c"
        if i == 1:
            subject_category_id = "d"
        mat_data = read_data()
        control_subjects = mat_data[category_str[i]]
        control_subjects_num = control_subjects.shape[1]
        print(control_subjects_num)

        user_indices = list(range(control_subjects_num))
        random.shuffle(user_indices)

        remaining_users = len(user_indices) % folder_num

        data_allocation(remaining_users, user_indices, control_subjects, subject_category_id, folder_num)


if __name__ == '__main__':
    deal_data()
