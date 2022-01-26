
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



def get_data():
    train = []
    test = []
    data = []
    vocab = set()
    with open('/kaggle/input/s21genetraintxt/S21-gene-train.txt') as f:
        for l in f.readlines():
            l = l.strip()
            if l != '' and l != '\n':
                t = l.split('\t')
                vocab.add(t[1])
                train.append(t)
            else:
                data.append(train)
                train = []
        if train:
            data.append(train)
                
    data = np.array(data, dtype=object)
    return data
