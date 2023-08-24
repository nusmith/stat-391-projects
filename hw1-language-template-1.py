#!/usr/bin/env python
import numpy as np
import string

# Read the language statistics

def readLangStats( filename ):
    # first version
    peng = np.zeros( 26, dtype = float )
    i = 0
    for line in open( filename ):
        dum = line.split( ' ' )
        pdum = float( dum[ 2 ] )/1000.
        peng[ i ] = pdum
        i = i+1
    #second and simpler version
    pengi = []

    for line in open( filename ):
        dum = line.split( ' ' )
        pdum = float( dum[ 2 ] )/1000.
        pengi.append( pdum )

    peng = np.array( pengi )
    return peng

def normalize( vec):
    svec = sum( vec )
    vec = vec / svec
    return None  #optional


# main 

peng = readLangStats('english-1.dat')
pger = readLangStats('german-1.dat')
pfr = readLangStats('french-1.dat')
psp = readLangStats('spanish-1.dat')
alphabet = 'abcdefghijklmnopqrstuvwxyz'
nletters = len(alphabet)
languages = ['English', 'German ', 'Spanish', 'French ' ] #space added to make them all equal length

while True:
    sentence = input('Enter a sentence:')

    # Preprocess: Turn all letters to lower case, eliminate spaces and punctuation
    sentence = sentence.lower()
    sentence = sentence.strip()
    sentence = sentence.translate((str.maketrans('','', string.punctuation)))

    # Get the sufficient statistics: Count the number of times each letter appears in the sentence. These
    # are the counts na, nb, . . . nz
    counts = [0] * 26
    for letter in alphabet:
        counts[alphabet.index(letter)] = sentence.count(letter)
    print(counts)
    # Compute and print the log-likelihood of the sentence in that language
    L_eng = 1
    L_ger = 1
    L_span = 1
    L_fran = 1
    for i in range(26):
        L_eng *= peng[i]**counts[i]
        L_ger *= pger[i]**counts[i]
        L_span *= psp[i]**counts[i]
        L_fran *= pfr[i]**counts[i]
    logL_eng = np.log2(L_eng)
    logL_ger = np.log2(L_ger)
    logL_span = np.log2(L_span)
    logL_fran = np.log2(L_fran)
    L = [L_eng, L_ger, L_span, L_fran]
    ll = [logL_eng, logL_ger, logL_span, logL_fran]

    # print results
    print('Log-likelihoods (in bits)')
    for i in ll:
        print(i)
    ilang = ll.index(max(ll))
    print('The most likely language of %s is %s' %(sentence, languages[ ilang ]))


    




