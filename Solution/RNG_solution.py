#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# From Wikipedia
# https://en.wikipedia.org/wiki/Linear_congruential_generator
# 
# A linear congruential generator (LCG) is an algorithm that yields a sequence of pseudo-randomized numbers calculated with a discontinuous piecewise linear equation. The method represents one of the oldest and best-known pseudorandom number generator algorithms. The theory behind them is relatively easy to understand, and they are easily implemented and fast, especially on computer hardware which can provide modulo arithmetic by storage-bit truncation.
# 
# The generator is defined by recurrence relation:
# 
# \begin{align}
# X_{n+1} = (aX_{n} + c)\:mod\:m
# \end{align}
# 
# where X is the sequence of pseudorandom values, and
# 
# \begin{align}
# m, 0<m - the\:modulus \\
# a, 0<a<m - the\:multiplier \\
# c, 0 \leqslant c<m - the\:increment \\
# X_{0}, 0 \leqslant X_{0} < m - the\:seed
# \end{align}
# 
# are integer constants that specify the generator.

# In[2]:


# helper function
def int_to_bin_seq(x, num_digit):
    '''
    converts integer to binary sequence list
    of specified length by num_digit
    
    e.g. x=4, num_digit=3  to [1,0,0]
    e.g. x=4, num_digit=4  to [0,1,0,0]
    '''
    # convert int to binary string
    # e.g. 4 to '100'
    x_bin = '{0:b}'.format(x)
    # convert binary string to sequence of integers
    # e.g. '100' to [1, 0 ,0]
    x_bin_seq = list(map(int,x_bin))
    
    # add zeros in front if needed
    x_bin_seq = [0]*(num_digit-len(x_bin_seq)) + x_bin_seq
    
    return x_bin_seq


# In[3]:


def lcg_generator(m, a, c, seq_len, seed = 'random'):
    '''
    For a given m and coeffs a and c,
    generates binary numbers of length (seq_len),
    based on linear congruential generator algorithm.
    '''
    # calculate the num of binary digits
    # necessary to represent nums.
    num_digit = int(np.ceil(np.log2(m)))
    
    # generate random seed
    if seed == 'random':
        x_init = np.random.randint(0,m)
    else:
        x_init = seed

    # convert int to binary sequence
    x_init_bin_seq = int_to_bin_seq(x_init, num_digit)

    # initialize sequence
    sequence = []
    sequence += x_init_bin_seq

    # initialize recurrence relation
    x = x_init
    
    while len(sequence) < seq_len:
        # recurrence relation
        x_next = (a*x + c) % m
        # convert int to binary sequence
        x_next_bin_seq = int_to_bin_seq(x_next, num_digit)
        # add this to sequence
        sequence += x_next_bin_seq
        # prepare for the following loop
        x = x_next
        
    # crop to fixed size    
    sequence = sequence[0:seq_len]
    
    return sequence


# In[4]:


RNs_training = lcg_generator(m=15, a=5, c=10, seq_len=2000, seed = 'random')
RNs_test = lcg_generator(m=15, a=5, c=10, seq_len=500, seed = 'random')


# In[5]:


def dataset_generator(RNs, inp_seq_len, step):
    X = []
    y = []
    i = 0
    while i+inp_seq_len+1<=len(RNs):
        X.append(RNs[i:i+inp_seq_len])
        y.append(RNs[i+inp_seq_len])
        i+=step
    return np.array(X), np.array(y)


# In[6]:


inp_seq_len = 50
X_train, y_train = dataset_generator(RNs_training, inp_seq_len, step=1)
X_test, y_test = dataset_generator(RNs_test, inp_seq_len, step=1)


# In[7]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


# In[8]:


model = Sequential()
model.add(Dense(20, input_dim=inp_seq_len, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[9]:


from keras.optimizers import Adam
opt = Adam()
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])


# In[10]:


H = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))


# In[11]:


f = open("JRNG.txt", "r") # open
JRNG = list(f.read()) # read
JRNG = list(map(int, JRNG)) # convert to integers
JRNG = np.array(JRNG) # convert list to np array


# In[12]:


# check if it is only 0's and 1's
np.unique(JRNG)


# In[13]:


# delete non-0's and non-1's
JRNG = JRNG[JRNG != 2]
JRNG = JRNG[JRNG != 4]
np.unique(JRNG)


# In[14]:


inp_seq_len = 50
X_train, y_train = dataset_generator(JRNG[0:1500], inp_seq_len, step=1)
X_test, y_test = dataset_generator(JRNG[1500:], inp_seq_len, step=1)


# In[15]:


model = Sequential()
model.add(Dense(20, input_dim=inp_seq_len, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[16]:


opt = Adam()
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])


# In[17]:


H = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

