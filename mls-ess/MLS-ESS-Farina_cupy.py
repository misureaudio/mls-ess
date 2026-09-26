import numpy as np
import soundfile as sf  # or use scipy.io.wavfile
import datetime
import os

from scipy.signal import max_len_seq

import cupy as cp
import cupyx.scipy.signal as signal  # ← this is the key import


def generate_mls_scipy(n_bits):
    """Generate Maximum Length Sequence (MLS) of length 2^n_bits - 1 using scipy"""
    seq, state = max_len_seq(n_bits)          # seq is 0/1 array
    mls_seq = 2 * seq.astype(int) - 1         # convert to bipolar ±1
    return mls_seq


def generate_mls_17_gr4(n_bits):
    if n_bits != 17:
        raise ValueError("Only implemented for n=17 with known primitive poly")

    length = 2**n_bits - 1
    register = np.ones(n_bits, dtype=int)          # initial all-1s state
    mls_seq = np.zeros(length, dtype=int)

    for i in range(length):
        # Standard primitive poly for n=17: x^17 + x^14 + 1
        # register[0] = newest, register[-1] = oldest
        feedback = register[-1] ^ register[3]      # bit 17 ⊕ bit 14 (0-based: -1=16, 3=bit14 if MSB=16)
        mls_seq[i] = register[-1]

        register[1:] = register[:-1]
        register[0] = feedback

    return 2 * mls_seq - 1


# -------------------------------------------------------------------
# ESS sequence 10*2^17 long
# -------------------------------------------------------------------
print('ESS begin', datetime.datetime.now())

# Working directory
wkdir = './'

# Sampling frequency
fs = 48000
nbit = 17
lMLS = 2**nbit - 1
noct = 10
lseq = noct * lMLS  # theoretical length (floating point)
nseq = int(round(lseq))  # actual length (integer)

# Generate ESS signal (vectorized)
xESS = np.arange(nseq)
noct22 = 2**noct

# ESS formula from the paper: sin((π/(2^N)) * L / log(2^N) * exp(x/N * log(2^N)))
yESS = np.sin((np.pi / noct22 * lseq) / np.log(noct22) * np.exp(xESS / nseq * np.log(noct22)))

# Write to WAV file (stereo by duplicating channel)
sf.write(os.path.join(wkdir, 'yESS.wav'), np.column_stack((yESS, yESS)), fs)
print('ESS end')

# -------------------------------------------------------------------
# Time reversal of the ESS signal
# -------------------------------------------------------------------
print('inv ESS begin', datetime.datetime.now())

# Vectorized implementation
yESS_rev = yESS[::-1]  # reverse the array
k2 = noct * np.log(2) / (1 - 1 / (2**noct))

# Create the exponential weighting factor vectorized
i_vec = np.arange(1, nseq + 1)
yESSinv = yESS_rev / (2**(noct / nseq * i_vec)) * k2

# Normalize
yESSinv = yESSinv / np.max(np.abs(yESSinv))

# Write to WAV file
sf.write(os.path.join(wkdir, 'yESSinv.wav'), np.column_stack((yESSinv, yESSinv)), fs)
print('inv ESS end')

# -------------------------------------------------------------------
# 41 MLS sequences 2^17-1 long
# -------------------------------------------------------------------
print('MLS begin', datetime.datetime.now())

nMLS = 41
yMLS = np.zeros((nMLS, lMLS))

# Generate MLS sequences
generate_mls = generate_mls_scipy
# generate_mls = generate_mls_17_gr4
for i in range(nMLS):
    yMLS[i, :] = generate_mls(nbit)

# Central sequence zeroed (MATLAB uses (nMLS-1)/2+1 for 1-based indexing)
center_idx = (nMLS - 1) // 2  # Python 0-based indexing
yMLS[center_idx, :] = 0

# Create final MLS sequence by concatenating
yMLSfin = np.zeros(nMLS * lMLS)
for i in range(nMLS):
    yMLSfin[i * lMLS:(i + 1) * lMLS] = yMLS[i, :]

# Write to WAV file
sf.write(os.path.join(wkdir, 'yMLSfin.wav'), np.column_stack((yMLSfin, yMLSfin)), fs)
print('MLS end')

# -------------------------------------------------------------------
# Convolution: MLS with ESS - very long time
# -------------------------------------------------------------------
'''
print('MLSESS convolution begin', datetime.datetime.now())

# Convolution
MLSESScnv = np.convolve(yMLSfin, yESS, mode='full')

# Normalize
MLSESScnv = MLSESScnv / np.max(np.abs(MLSESScnv))

# Write full file with tails
sf.write(os.path.join(wkdir, 'MLSESScnv.wav'), MLSESScnv, fs)
print('MLSESS convolution end')

print('MLSESS tail cut begin', datetime.datetime.now())

# Signal length with both tails
lMLSESScnv = len(MLSESScnv)

# Tail length
ltail = nseq

# Read refreshed MLSESScnv signal with no tails (simulate reading from file)
# In Python, we can directly slice the array instead of reading from file
MLSESScnv_no_tails = MLSESScnv[ltail:lMLSESScnv - ltail]
'''

# ────────────────────────────────────────────────
print('MLSESS convolution begin (GPU)', datetime.datetime.now())

# Move signals to GPU (they must be CuPy arrays)
yMLSfin_gpu = cp.asarray(yMLSfin)
yESS_gpu = cp.asarray(yESS)

# FFT convolution — this is usually much faster for long signals
MLSESScnv_gpu = signal.fftconvolve(yMLSfin_gpu, yESS_gpu, mode='full')

# Move result back to CPU/RAM for saving + further slicing
MLSESScnv = cp.asnumpy(MLSESScnv_gpu)

# Normalize (on CPU is fine, or do it on GPU before transfer)
MLSESScnv = MLSESScnv / np.max(np.abs(MLSESScnv))

sf.write(os.path.join(wkdir, 'MLSESScnv.wav'), MLSESScnv, fs)
print('MLSESS convolution end (GPU)')

# Tail cutting remains the same (now on CPU)
lMLSESScnv = len(MLSESScnv)
ltail = nseq
MLSESScnv_no_tails = MLSESScnv[ltail:lMLSESScnv - ltail]
# sf.write(os.path.join(wkdir, 'MLSESScnv_notails.wav'), MLSESScnv_no_tails, fs)

# Save MLSESScnv file without tails
sf.write(os.path.join(wkdir, 'MLSESScnv_notails.wav'), MLSESScnv_no_tails, fs)
print('MLSESS tail cut end', datetime.datetime.now())
