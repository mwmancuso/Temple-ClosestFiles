#!/usr/bin/env python

#
# Author: Matt Mancuso
# Date: 4/24/19
#

import itertools
import multiprocessing
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import hashlib

mmh3_ready = False
try:
    import mmh3

    mmh3_ready = True
except ImportError:
    pass

# SETTINGS
# Seed is for reproducibility
# ngrams are length of ngrams to produce from files
# hash_len is number of minhashes from ngrams to take
# hash_chunk is number of minhashes to sum for individual
#   hash comparisons. thus hash_len/hash_chunk is how many
#   chances two files get to be similar based on the
#   hash_len random ngrams calculated
random.seed(1036)
ngrams = 2
ngrams_final_cmp = 1
hash_len = 32
hash_initial_divider = 4

# Generate 128 bit hash seeds to XOR with MD5 hashes. need
# one for each hash
hashseeds = []
for i in range(hash_len):
    hashseeds.append(random.randint(2 ** 127, 2 ** 128 - 1))

# We match words by any word character (A-Z, a-z, 0-9, ' or _)
# surrounded by non word characters (beginning/end of string),
# whitespace, dashes, periods, etc.
words_re = re.compile(r'\b([a-z0-9\-_\']+)\b', re.MULTILINE | re.IGNORECASE)


def min_hash(file):
    """
    Calculates hash_len min hashes for each input
    set of ngrams. Each minhash will be different because
    hashseeds has 128-bit seeds which will be XORd with
    the hash.
    :param f: set of hashed ngrams for a file
    :return: list of hash_len minhashes
    """
    try:
        f = open(file)
    except IOError:
        return [0 for i in range(hash_len)]

    # Findall will use above regex to match any words.
    # We also convert all words to lowercase and reduce all
    # newlines to spaces to create one long string.
    words = words_re.findall(f.read())

    f.close()

    # Create ngrams, hash them, and return them as a set
    if mmh3_ready:
        f = set([mmh3.hash(' '.join(words[i:i + ngrams])) for i in range(0, len(words) - ngrams + 1)])
    else:
        f = set([int(hashlib.md5(' '.join(words[i:i + ngrams])).hexdigest(), 16)
                 for i in range(0, len(words) - ngrams + 1)])

    signature = []

    if len(f) == 0:
        return [0 for i in range(hash_len)]

    signature.append(min(f))

    for i in range(hash_len - 1):
        # Get a 128-bit seed and XOR all hashes with 'em,
        # then add the minimum hash to the output
        seed = hashseeds[i]
        signature.append(min([v ^ seed for v in f]))

    return signature


def diff_index((f1, f2)):
    """
    Calculates difference index between two files. The inputs
    f1 and f2 must be sets, but can be either sets of ngrams
    or sets of ngram hashes. Sets of ngrams will be more
    accurate. The algorithm takes the number of the same
    elements divided by the total number of elements.

    :return: int, how many disagreeing trigrams
    """
    try:
        file1 = open(f1)
        file2 = open(f2)
    except IOError:
        return 1000000

    # Findall will use above regex to match any words.
    # We also convert all words to lowercase and reduce all
    # newlines to spaces to create one long string.
    words1 = words_re.findall(file1.read())
    words2 = words_re.findall(file2.read())

    file1.close()
    file2.close()

    # Create ngrams, hash them, and return them as a set
    if mmh3_ready:
        f1 = Counter([mmh3.hash(' '.join(words1[i:i + ngrams_final_cmp])) for i in range(0, len(words1) - ngrams_final_cmp + 1)])
        f2 = Counter([mmh3.hash(' '.join(words2[i:i + ngrams_final_cmp])) for i in range(0, len(words2) - ngrams_final_cmp + 1)])
    else:
        f1 = Counter([int(hashlib.md5(' '.join(words1[i:i + ngrams_final_cmp])).hexdigest(), 16)
                 for i in range(0, len(words1) - ngrams_final_cmp + 1)])
        f2 = Counter([int(hashlib.md5(' '.join(words2[i:i + ngrams_final_cmp])).hexdigest(), 16)
                 for i in range(0, len(words2) - ngrams_final_cmp + 1)])

    return sum([v for k,v in (f1 - f2).items()]) + sum([v for k,v in (f2 - f1).items()])


def main(norm_argv):
    files = []
    pool = multiprocessing.Pool()

    print "> starting"

    if mmh3_ready:
        print "> using MMH3 for hashes"
    else:
        print "> using MD5 for hashes"

    print "> listing files..."

    t_total = time.time()

    # Get list of files first and foremost
    for root, directories, fs in os.walk(norm_argv[0]):
        files.extend([os.path.join(root, fname) for fname in fs if fname[-4:] == '.txt'])

    print "> listed " + str(len(files)) + " files"

    if len(files) < 2:
        print "! need two or more files. aborting"
        return

    print "> using " + str(ngrams) + "grams for minhashes..."

    print "- elapsed time: " + str(time.time() - t_total)

    print "> generating minhash(" + str(hash_len) + ")..."

    # Calculate the minhash signature for each file.
    # signatures will be a list equal in length to
    # the number of files. Each element will be
    # another list of hash_len/hash_chunk length
    # which holds the minhash signature of the file
    signatures = pool.map(min_hash, files)

    print "> minhashes done"
    print "- elapsed time: " + str(time.time() - t_total)

    print "> finding optimal chunk size..."

    # We need to find the best hash_chunk size, i.e.
    # the hash_chunk that yields the fewest comparisons
    # possible, i.e. the biggest hash_chunk size.
    # To do this, we start with hash_chunk==hash_len
    # and sum all of the hashes together to compare.
    # If that doesn't yield any matches (i.e. two
    # files that share the same hash), we cut
    # hash_chunk in half and sum each half, then
    # compare again. We keep doing this until we get
    # at least two files which share a hash in common.
    # If hash_chunk gets to 0, we know hash_len is too
    # short to find similar files, and we abort.
    related_files_idxs = []
    hash_chunk = hash_len / hash_initial_divider
    while hash_chunk > 0 and len(related_files_idxs) == 0:
        print "> testing hash_chunk of " + str(hash_chunk)

        # We take every hash from every signature and add
        # them to a dictionary, where the value is a list
        # of files that have that hash. We can thus find
        # any files that are similar by finding any hash
        # with more than one file. The hashes also must
        # appear at the same index in the signature to
        # be considered similar.
        all_hashes = dict()
        for i, fname in enumerate(files):
            for c in range(len(signatures[i]) / hash_chunk):
                sidx_beg = c * hash_chunk
                sidx_end = (c + 1) * hash_chunk
                sig_hash = sum(signatures[i][sidx_beg:sidx_end])
                if (c, sig_hash) in all_hashes:
                    all_hashes[(c, sig_hash)].append(i)
                else:
                    all_hashes[(c, sig_hash)] = [i]

        # Related files are any files who appear together
        # in the all_hashes dict. Thus related_files_idxs
        # will be a list of tuples where each item in the
        # tuple is similar to other items in the tuple
        related_files_idxs = [v for v in all_hashes.itervalues() if len(v) > 1]

        if len(related_files_idxs) == 0:
            hash_chunk /= 2

    print "> settled on hash_chunk of " + str(hash_chunk)

    for j in [0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.995]:
        print "  ? probability of finding files with " + str(j) + " jaccard index: " \
              + str(1.0 - (1.0 - j ** hash_chunk) ** (hash_len / hash_chunk))

    print "- elapsed time: " + str(time.time() - t_total)

    if len(related_files_idxs) == 0:
        print "! no files close enough to analyze. aborting"
        exit()

    print "> analyzing similar hashes..."

    print "> " + str(len(related_files_idxs)) + " hashes matched between at least two files"

    if len(related_files_idxs) > 62500:
        print "   ? that's a lot, this might take awhile"

    # Now that we have a big list of tuples where the
    # tuples hold files which are related, we can
    # generate combinations of the tuples to build a
    # list of all the comparisons we need to make.
    # file_matches = set()
    # for related_files in related_files_idxs:
    #     file_matches.update(itertools.combinations(sorted(related_files), 2))
    #
    # The below code is practically unreadable but does
    # the same thing as above, slightly faster:
    file_matches = list(
        set([c for related_files in related_files_idxs for c in itertools.combinations(sorted(related_files), 2)]))

    # Map the indices of the file ngrams to the set
    # of ngrams themselves. We have to do this in case
    # the parallel pool is used.
    comparisons = [(files[c[0]], files[c[1]]) for c in file_matches]

    print "> comparing " + str(len(comparisons)) + " similar files..."
    print "> using " + str(ngrams_final_cmp) + "grams for comparisons..."

    if len(comparisons) > 500000:
        print "   ? that's a lot, this might take awhile"

    # Calculate the number of different words for each pair of
    # files. We do this in a pool because it requires significant
    # I/O per file
    difference_indices = pool.map(diff_index, comparisons)

    # Get the maximum Jaccard index and its associated
    # file indices.
    min_difference = min(difference_indices)
    closest_index = difference_indices.index(min_difference)
    closest_files = file_matches[closest_index]
    # jaccard_index = float(len(comparisons[closest_index][0] & comparisons[closest_index][1])) \
    #                 / float(len(comparisons[closest_index][0] | comparisons[closest_index][1]))

    print "> compared " + str(len(comparisons)) + " similar files"
    print "> average difference for similar files: " + str(np.mean(difference_indices))
    print "- total run time: " + str(time.time() - t_total)
    print "> run time per file: " + str(float(time.time() - t_total) / float(len(files)))
    print "> est. run time for 1,000,000 files: " + str(1000000.0 * float(time.time() - t_total) / float(len(files)))

    # Finally, print the files:
    # print "> jaccard index of two closest files: " + str(jaccard_index)
    print "> difference of two closest files: " + str(min_difference)
    print "> two closest files are:"
    print files[closest_files[0]]
    print files[closest_files[1]]


def show_help():
    print """\
name: compute_closest.py
synopsis: compute_closest.py [directory of files]
descr: finds two most similar files in database of text files
example: compute_closest.py data/

options:
 -help   --help: display this help message
 
arguments:
 directory of files: a directory with any number of nested
     .txt files
     
man page: none\
"""
    exit()


def check_args(argv):
    if len(argv) != 1:
        show_help()

    if not os.path.isdir(argv[0]):
        print "bad directory"
        exit()

    return argv


if __name__ == "__main__":
    main(check_args(sys.argv[1:]))
