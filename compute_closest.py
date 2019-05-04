#!/usr/bin/env python

#
# Author: Matt Mancuso
# Date: 4/24/19
#

import hashlib
import itertools
import multiprocessing
import os
import random
import re
import sys
import time
from collections import Counter
import numpy as np

# SETTINGS
# Seed is for reproducibility
# ngrams are length of ngrams to produce from files
# hash_len is number of minhashes from ngrams to take
# hash_chunk is number of minhashes to sum for individual
#   hash comparisons. thus hash_len/hash_chunk is how many
#   chances two files get to be similar based on the
#   hash_len random ngrams calculated
random.seed(1235)
ngrams = 3
hash_len = 32
hash_chunk = 8

# Generate 128 bit hash seeds to XOR with MD5 hashes. need
# one for each hash
hashseeds = []
for i in range(hash_len):
    hashseeds.append(random.randint(2 ** 127, 2 ** 128 - 1))

# We match words by any word character (A-Z, a-z, 0-9 or _)
# surrounded by non word characters (beginning/end of string),
# whitespace, dashes, periods, etc.
words_re = re.compile(r'\b([a-z0-9\-_]+)\b', re.MULTILINE | re.IGNORECASE)


def grams_per_file(file):
    """
    Get all ngrams for file. Returns set of distinct
    ngrams, dependent on ngrams specified in header.
    :param file: full filepath to file to analyze
    :return: set of distinct ngrams in file
    """

    try:
        f = open(file)
    except IOError:
        return set()

    # Findall will use above regex to match any words.
    # We also convert all words to lowercase and reduce all
    # newlines to spaces to create one long string.
    words = words_re.findall(f.read())

    f.close()

    # Create ngrams and return them as a set
    return set([' '.join(words[i:i + ngrams]) for i in range(0, len(words) - ngrams + 1)])


def min_hash(f):
    """
    Calculates hash_len/hash_chunk min hashes for each input
    set of ngrams. Each minhash will be different because
    hashseeds has 128-bit seeds which will be XORd with
    the hash.
    :param f: set of ngrams for a file
    :return: list of hash_len/hash_chunk minhashes
    """
    signature = []

    if len(f) == 0:
        return []

    # We calculate the MD5 sum for each ngram only once.
    # This saves A LOT of time. We'll seed the hashes
    # with XOR instead of inputs to MD5
    hashes = map(lambda s: int(hashlib.md5(s).hexdigest(), 16), list(f))

    for i in range(hash_len):
        # Get a 128-bit seed and XOR all of the hashes
        # with them
        seed = hashseeds[i]
        seeded_hashes = map(lambda v: v ^ seed, hashes)

        # To combine hash_chunk hashes, we sum them up
        # into a single index of the signature.
        # Note that we are taking the MIN of the seeded
        # hashes, hence minhash. This will change
        # depending on the seed, and we will thus get
        # a random ngram for each index in the signature
        if i % hash_chunk == 0:
            signature.append(min(seeded_hashes))
        else:
            signature[i / hash_chunk] += min(seeded_hashes)

    return signature


def jaccard_index((f1, f2)):
    """
    Calculates Jaccard index between two files. The inputs
    f1 and f2 must be sets, but can be either sets of ngrams
    or sets of ngram hashes. Sets of ngrams will be more
    accurate. The algorithm takes the number of the same
    elements divided by the total number of elements.

    :return: a float where 0 is nothing in common, 1 is exact match
    """

    return float(len(f1 & f2)) / float(len(f1 | f2))


def main(norm_argv):
    files = []
    pool = multiprocessing.Pool()

    print "> starting"
    print "> listing files..."

    t_total = time.time()

    # Get list of files first and foremost
    for root, directories, fs in os.walk(norm_argv[0]):
        files.extend([os.path.join(root, fname) for fname in fs if fname[-4:] == '.txt'])

    print "> listed " + str(len(files)) + " files"

    if len(files) < 2:
        print "! need two or more files. aborting"
        return

    print "> generating " + str(ngrams) + "grams..."

    # Calculate the ngrams for each file. file_ngrams
    # will be a list of sets of ngrams for each file
    file_ngrams = pool.map(grams_per_file, files)

    print "> " + str(ngrams) + "grams done"
    print "- elapsed time: " + str(time.time() - t_total)

    print "> generating minhash(" + str(hash_chunk) + "/" + str(hash_len) + ")..."

    # Calculate the minhash signature for each file.
    # signatures will be a list equal in length to
    # the number of files. Each element will be
    # another list of hash_len/hash_chunk length
    # which holds the minhash signature of the file
    signatures = pool.map(min_hash, file_ngrams)

    print "> minhashes done"
    print "- elapsed time: " + str(time.time() - t_total)

    print "> analyzing similar hashes..."

    for j in [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
        print "  ? probability of finding files with " + str(j) + " jaccard index: " \
              + str(1.0 - (1.0 - j ** hash_chunk) ** (hash_len / hash_chunk))

    # We take every hash from every signature and add
    # them to a dictionary, where the value is a list
    # of files that have that hash. We can thus find
    # any files that are similar by finding any hash
    # with more than one file. The hashes also must
    # appear at the same index in the signature to
    # be considered similar.
    all_hashes = dict()
    for i, fname in enumerate(files):
        for sidx, sig_hash in enumerate(signatures[i]):
            if (sidx, sig_hash) in all_hashes:
                all_hashes[(sidx, sig_hash)].append(i)
            else:
                all_hashes[(sidx, sig_hash)] = [i]

    # Related files are any files who appear together
    # in the all_hashes dict. Thus related_files_idxs
    # will be a list of tuples where each item in the
    # tuple is similar to other items in the tuple
    related_files_idxs = filter(lambda v: len(v) > 1, all_hashes.values())

    if len(related_files_idxs) == 0:
        print "! no files close enough to analyze. aborting"
        exit()

    # Now that we have a big list of tuples where the
    # tuples hold files which are related, we can
    # generate combinations of the tuples to build a
    # list of all the comparisons we need to make.
    # We'll also be able to determine files which match
    # for more than one hash, and can thus use that
    # metric to further reduce the number of comparisons
    # we need to make.
    file_matches = list()
    for related_files in related_files_idxs:
        file_matches.extend(itertools.combinations(sorted(related_files), 2))

    # We throw all of the comparisons into a counter,
    # so we know which files have more than one chunk
    # of the signature in common.
    counts = Counter()
    for file_match in file_matches:
        counts[file_match] += 1

    # Yield the max number of chunks for two files in
    # common. We use this to filter only the most
    # common files.
    max_count = max(counts.values())

    # If we have any files with more than one chunk
    # matching, we can filter the list of files to
    # compare by the max number of matching chunks
    # calculated above. This will reduce the number
    # of comparisons we need to make significantly,
    # as for very similar files, most of the chunks
    # will match.
    if max_count > 1:
        file_matches = map(lambda (k, v): k, filter(lambda (k, v): v == max_count, counts.items()))

    # Map the indices of the file ngrams to the set
    # of ngrams themselves. We have to do this in case
    # the parallel pool is used.
    comparisons = map(lambda c: (file_ngrams[c[0]], file_ngrams[c[1]]), file_matches)

    # Calculate the Jaccard indices of all file combos.
    # Parallel pools are very resource intensive to
    # start up, so only invoke them if we have a large
    # number of comparisons to make. Otherwise, just
    # calculate all Jaccard indices using a single
    # thread. The higher the Jaccard index, the closer
    # the files.
    if len(comparisons) > 50000:
        jaccard_indices = pool.map(jaccard_index, comparisons)
    else:
        jaccard_indices = map(jaccard_index, comparisons)

    # Get the maximum Jaccard index and its associated
    # file indices.
    max_jaccard = max(jaccard_indices)
    closest_files = file_matches[jaccard_indices.index(max_jaccard)]

    print "> compared " + str(len(comparisons)) + " similar files"
    print "> average jaccard index for similar files: " + str(np.mean(jaccard_indices))
    print "- total run time: " + str(time.time() - t_total)
    print "> run time per file: " + str(float(time.time() - t_total) / float(len(files)))
    print "> est. run time for 1,000,000 files: " + str(1000000.0 * float(time.time() - t_total) / float(len(files)))

    # Finally, print the files:
    print "> jaccard index of two closest files: " + str(max_jaccard)
    print "> two closest files are:"
    print files[closest_files[0]]
    print files[closest_files[1]]


def chunks(l, n):
    """
    Get a list of lists of size n, thus the size returned
    will be ret[ceil(l/n)][n].

    :param l: original list
    :param n: n in ret[ceil(l/n)][n]
    :return: list of lists of size n
    """
    n = max(1, n)
    return [l[i:min(i + n, len(l))] for i in xrange(0, len(l), n)]


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
