#!/usr/bin/env python

#
# Author: Matt Mancuso
# Date: 4/24/19
#

import multiprocessing
import sys
import os
import re
import time
from collections import Counter
from math import ceil
import numpy as np


def grams_per_file(files):
    """
    Get total word counts for file per file. Returns list of equal
    size of input where each item is a dict of words.
    :param files: list of files with full path
    :return: list of dicts with word counts per file
    """

    # We match words by any word character (A-Z, a-z, 0-9 or _)
    # surrounded by non word characters (beginning/end of string),
    # whitespace, dashes, periods, etc.
    words_re = re.compile(r'\b(\w+)\b')

    # A Counter object has special operators to make adding/
    # combining dicts trivial
    ret = list()

    # Loop through all files
    for file in files:
        wordcounts = Counter()

        try:
            f = open(file)
        except IOError:
            continue

        # Findall will use above regex to match any words.
        # We also convert all words to lowercase and reduce all
        # newlines to spaces to create one long string.
        words = words_re.findall(f.read().lower().replace('\n', ' '))

        grams = [' '.join(words[i:i + 3]) for i in range(0, len(words) - 3 + 1)]

        # Build the wordcounts dict by either creating key or
        # adding to it
        for gram in grams:
            wordcounts[gram] += 1

        f.close()

        ret.append(wordcounts)

    return ret


def jaccard_index(f1, f2):
    """

    :param f1:
    :param f2:
    :return:
    """

    return 1.0 - (float(sum([v for k,v in (f1 - f2).items()])
                        + sum([v for k,v in (f2 - f1).items()]))
                  / float(sum([v for k,v in (f1 + f2).items()])))


def main(norm_argv):
    files = []
    pool = multiprocessing.Pool()

    t = time.time()

    # Get list of files first and foremost
    for root, directories, fs in os.walk(norm_argv[0]):
        files.extend([os.path.join(root, fname) for fname in fs])

    # Number of CPUs is number of lists we need for pool map
    cpus = multiprocessing.cpu_count()

    # We split the list of files into k chunks of size n, where k
    # is number of cores (cpus) and n is ceil(#files/k), to yield
    # cpus lists of near-equal size.
    cs = chunks(files, int(ceil(len(files)/float(cpus))))

    # Run file_get_wc on each list in cs, thus one per core.
    # Pool will take care of spawning processes and running it
    ret = pool.map(grams_per_file, cs)

    # Counter object can be summed to add together separate
    # dicts (one for each CPU). Result will be intersection
    # of dictionaries where keys that occur in multiple dicts
    # are summed by value
    r = sum(ret, [])

    diffs = np.ones((len(r), len(r)))*(-np.inf)
    # Print run time and top words
    print "run time: " + str(time.time() - t)


    t = time.time()
    for i in range(len(r) - 1):
        print i
        for c in range(i + 1, len(r)):
            diffs[i,c] = jaccard_index(r[i], r[c])

    print "run time: " + str(time.time() - t)

    (xinds, yinds) = np.where(diffs == diffs.max())
    print diffs.max()
    for ind in range(len(xinds)):
        xind = xinds[ind]
        yind = yinds[ind]
        print files[xind] + " - " + files[yind]

    print diffs


def chunks(l, n):
    """
    Get a list of lists of size n, thus the size returned
    will be ret[ceil(l/n)][n].

    :param l: original list
    :param n: n in ret[ceil(l/n)][n]
    :return: list of lists of size n
    """
    n = max(1, n)
    return [l[i:min(i+n, len(l))] for i in xrange(0, len(l), n)]


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
