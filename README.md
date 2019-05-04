# Temple-ClosestFiles
Code that uses MinHashing to compute the two most similar files in ~O(n) time.

Inspiration:
https://medium.com/engineering-brainly/locality-sensitive-hashing-explained-304eb39291e4

How it Works:
1. Generate all of the trigrams in all of the files
2. For each file, calculate the min hashes:
   1. First, we need some parameters:
      - number of ngrams to hash (hash_len): 32
      - number of ngram hashes to combine (hash_chunk):
        determined in step 3
   2. We then need to generate a random seed for each
      ngram hash (thus we need 32). Because we're using
      MD5 and want as much entropy as possible, we generate
      128-bit numbers only. We'll use these seeds in step 4
   3. Hash all of the trigrams in the file with MD5
   4. XOR all of the MD5 hashes with the first seed
   5. Take the minimum hash numerically and append it to
      the signature list (the return value)
   6. Seed the MD5 hash from step 3 with a new seed
   7. Add the new minimum hash to the signature list.
   8. Repeat for all hash_len.
   9. We'll get a signature that looks like:
      [...123409L, ...234142L, ...344123L, ...388423L]
3. Calculate the ideal hash_chunk size by starting with
   hash_chunk == hash_len/4. Then go to step 4 and see if
   we succeed finding a list of files > 1 in length.
   If we can't, we reduce hash_chunk by half. Then,
   to create hash chunks, we simply sum hash_chunk hashes
   for each signature index. For hash chunk 8, hashes 0-7
   are summed for signature[0], 8-15 for signature[1], and
   so on.
4. We combine all hashes and their index into a
   dictionary, where the values for the hashes are a list
   of files which contain the hash in the same index in
   the signature. If the length of the list for any hash
   is > 1, we know we have at least one match for the
   two files, meaning they are fairly similar. Adjusting
   the parameters from step 2.1 and the ngram length (3
   for trigrams) can help pine down the algorithm to
   yield at least one match for more or less similar
   files.
5. Taking the combinations of the dictionary values will
   give us a list of files we need to compare for precise
   accuracy
6. Finally, take the list of file pairs and compare them
   using the Jaccard index on the set of trigrams.
   The Jaccard index is simply the number of trigrams
   present in both files divided by the total number of
   trigrams between the two files.
7. The closest files have the highest Jaccard index.
