from voice_svm import experiment_svm
import sys
from scipy.io.wavfile import write
import numpy as np

if len(sys.argv) < 2:
    print('\n Usage -> python voice-svm <wav file path>\n')
    sys.exit(1)

if __name__ == '__main__':
    wav_file = sys.argv[1]
    x_2, sample_rate = experiment_svm(wav_file)
    write("inverse.wav", sample_rate, x_2.astype(np.int16))