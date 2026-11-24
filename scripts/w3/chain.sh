#!/bin/bash
# w3 fit chain (pod, one fit at a time): 9B unstated 128 -> 2B unstated 248 -> 2B unstated wrong-passage control 64
S=/workspace/o-lens-improvement/scripts/w3/fit.sh
JA1_RESUME_EVERY=8 POSTN="64 96 128" bash $S 9b uns unstated 128
JA1_RESUME_EVERY=8 POSTN="64 128 248" bash $S 2b uns unstated 248
JA1_RESUME_EVERY=8 POSTN="64" bash $S 2b unswrong unstated 64 --wrong
