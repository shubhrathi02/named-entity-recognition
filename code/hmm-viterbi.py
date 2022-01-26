{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3e837e8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "323f9a40",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[['1', 'Comparison', 'O'], ['2', 'with', 'O'], ['3', 'alkaline', 'B'], ['4', 'phosphatases', 'I'], ['5', 'and', 'O'], ['6', '5', 'B'], ['7', '-', 'I'], ['8', 'nucleotidase', 'I'], ['9', '.', 'O']]\n",
      "13796\n",
      "[['1', 'To', 'O'], ['2', 'understand', 'O'], ['3', 'the', 'O'], ['4', 'molecular', 'O'], ['5', 'regulation', 'O'], ['6', 'of', 'O'], ['7', 'these', 'O'], ['8', 'genes', 'O'], ['9', 'in', 'O'], ['10', 'thyroid', 'O'], ['11', 'cells', 'O'], ['12', ',', 'O'], ['13', 'the', 'O'], ['14', 'effect', 'O'], ['15', 'of', 'O'], ['16', 'thyroid', 'B'], ['17', 'transcription', 'I'], ['18', 'factor', 'I'], ['19', '1', 'I'], ['20', '(', 'O'], ['21', 'TTF', 'B'], ['22', '-', 'I'], ['23', '1', 'I'], ['24', ')', 'O'], ['25', 'and', 'O'], ['26', 'the', 'O'], ['27', 'paired', 'B'], ['28', 'domain', 'I'], ['29', '-', 'I'], ['30', 'containing', 'I'], ['31', 'protein', 'I'], ['32', '8', 'I'], ['33', '(', 'O'], ['34', 'Pax', 'B'], ['35', '-', 'I'], ['36', '8', 'I'], ['37', ')', 'O'], ['38', 'on', 'O'], ['39', 'the', 'O'], ['40', 'transcriptional', 'O'], ['41', 'activity', 'O'], ['42', 'of', 'O'], ['43', 'the', 'O'], ['44', 'deiodinase', 'B'], ['45', 'promoters', 'I'], ['46', 'were', 'O'], ['47', 'studied', 'O'], ['48', '.', 'O']]\n",
      "31328\n"
     ]
    }
   ],
   "source": [
    "train = []\n",
    "test = []\n",
    "data = []\n",
    "vocab = set()\n",
    "with open('S21-gene-train.txt') as f:\n",
    "    for l in f.readlines():\n",
    "        l = l.strip()\n",
    "        if l != '' and l != '\\n':\n",
    "            t = l.split('\\t')\n",
    "            vocab.add(t[1])\n",
    "            train.append(t)\n",
    "        else:\n",
    "            data.append(train)\n",
    "            train = []\n",
    "    if train:\n",
    "        data.append(train)\n",
    "            \n",
    "data = np.array(data, dtype=object)\n",
    "print(data[0])\n",
    "print(len(data))\n",
    "print(data[13795])\n",
    "print(len(vocab))\n",
    "#print(.shape)\n",
    "#X_train = train[:,1]\n",
    "#y_train = train[:,2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "fea8f9ed",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-1.26478635 -4.13984761 -0.02430367]\n",
      " [-4.22115332 -0.23734668 -0.3758692 ]\n",
      " [-4.38806566 -0.21771592 -0.40471005]\n",
      " [-1.33688866 -5.53798523 -0.03904368]]\n"
     ]
    }
   ],
   "source": [
    "# Column/Row Sequence is B, I, O\n",
    "def get_index(c):\n",
    "    if c == 'B':\n",
    "        return 0\n",
    "    elif c == 'I':\n",
    "        return 1\n",
    "    else:\n",
    "        return 2\n",
    "\n",
    "\n",
    "b_prob = np.zeros((3, len(vocab)), dtype='float')\n",
    "a_prob = np.zeros((4, 3), dtype='float')\n",
    "vocab_index = dict()\n",
    "\n",
    "for i,v in enumerate(vocab):\n",
    "    vocab_index[v] = i\n",
    "\n",
    "\n",
    "count = np.zeros(4, dtype=int)\n",
    "count[0] = len(data) #start tag count\n",
    "\n",
    "for row in data:\n",
    "    for i, word in enumerate(row):\n",
    "        tag_index = get_index(word[2])\n",
    "        count[tag_index+1] += 1\n",
    "        if i==0:\n",
    "            a_prob[0][tag_index] += 1\n",
    "        else:\n",
    "            a_prob[get_index(row[i-1][2])+1][tag_index] +=1\n",
    "        \n",
    "        b_prob[tag_index][vocab_index[word[1]]] += 1\n",
    "        \n",
    "for i in range(4):\n",
    "    for j in range(3):\n",
    "        a_prob[i][j] = (a_prob[i][j]+1)/(count[i]+3)\n",
    "        a_prob[i][j] = np.log10(a_prob[i][j])\n",
    "\n",
    "vocab_size = len(vocab)\n",
    "for i in range(3):\n",
    "    for j in range(vocab_size):\n",
    "        b_prob[i][j] = (b_prob[i][j]+1)/(count[i]+vocab_size)\n",
    "        b_prob[i][j] = np.log10(b_prob[i][j])\n",
    "\n",
    "print(a_prob)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "196bd3dd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-143073.07485857952\n"
     ]
    }
   ],
   "source": [
    "print(sum(b_prob[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "53182d7d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-4.746346130376863\n",
      "-0.556070473819902\n"
     ]
    }
   ],
   "source": [
    "print(np.min(b_prob))\n",
    "print(np.max(b_prob))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "38c5f9ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_b_probs(i, w):\n",
    "    if w in vocab_index:\n",
    "        return b_prob[i][vocab_index[w]]\n",
    "    else:\n",
    "        print('Unknown')\n",
    "        if i == 2:\n",
    "            return 0\n",
    "        else:\n",
    "            return -10 #Very low probability\n",
    "    \n",
    "def viterbi(words):\n",
    "    t = len(words)\n",
    "    n = 3\n",
    "    prob_matrix = np.zeros((n, t), dtype=float)\n",
    "    back_pointer = np.zeros((n, t), dtype=int)\n",
    "    \n",
    "    for i in range(n):\n",
    "        prob_matrix[i, 0] = a_prob[0][i] + get_b_probs(i, words[0])\n",
    "        \n",
    "    for j in range(1, t):\n",
    "        for i in range(n):\n",
    "            max_prob = float('-inf')\n",
    "            max_arg = 0\n",
    "            for k in range(n):\n",
    "                p = prob_matrix[k][j-1] + a_prob[k+1][i]\n",
    "                if max_prob < p:\n",
    "                    max_prob = p\n",
    "                    max_arg = k\n",
    "                if j==5:\n",
    "                    print(max_prob)\n",
    "            prob_matrix[i][j] = max_prob + get_b_probs(i, words[j])\n",
    "            back_pointer[i][j] = max_arg\n",
    "    \n",
    "    ans = np.zeros(t, dtype=int)\n",
    "    ans[t-1] = np.argmax(prob_matrix[:, (t-1)])\n",
    "    \n",
    "    print(prob_matrix)\n",
    "    print(back_pointer)\n",
    "    for j in range(t-2, -1, -1):\n",
    "        ans[j] = back_pointer[ans[j+1], j+1]\n",
    "    return ans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "bcae636c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-23.420961621505008\n",
      "-20.203069713982078\n",
      "-15.337486584024402\n",
      "-19.437154979526593\n",
      "-16.03271997484289\n",
      "-16.03271997484289\n",
      "-19.575677495157016\n",
      "-16.21971409965029\n",
      "-14.039641603494635\n",
      "[2 2 2 2 2 2 2 2 2]\n"
     ]
    }
   ],
   "source": [
    "print(viterbi(['Comparison', 'with', 'alkaline', 'phosphatases', 'and', '5', '-', 'nucleotidase', '.']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "4ec04a8d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-4.65440759 -4.65440759 -3.37565399 -4.65440759 -4.65440759 -3.23943424\n",
      "  -4.17728634 -4.65440759 -4.65440759]\n",
      " [-4.68092445 -4.68092445 -3.63953176 -3.68092445 -2.55059068 -2.746426\n",
      "  -1.0380644  -4.37989445 -2.46081636]\n",
      " [-3.13356227 -1.21897405 -4.44531613 -4.26922488 -0.75304219 -1.80582965\n",
      "  -0.80404038 -4.44531613 -0.56035094]]\n"
     ]
    }
   ],
   "source": [
    "indexes = []\n",
    "indexes.append(vocab_index['Comparison'])\n",
    "indexes.append(vocab_index['with'])\n",
    "indexes.append(vocab_index['alkaline'])\n",
    "indexes.append(vocab_index['phosphatases'])\n",
    "indexes.append(vocab_index['and'])\n",
    "indexes.append(vocab_index['5'])\n",
    "indexes.append(vocab_index['-'])\n",
    "indexes.append(vocab_index['nucleotidase'])\n",
    "indexes.append(vocab_index['.'])\n",
    "\n",
    "print(b_prob[:,indexes])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bdda8ca6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
