#!/usr/bin/env bash
### create_submission.sh ---
##
## Filename: create_submission.sh
## Author: Fred Qi
## Created: 2017-05-30 15:35:46(+0800)
##
## Last-Updated: 2017-05-30 16:12:06(+0800) [by Fred Qi]
##     Update #: 18
######################################################################
##
### Commentary:
##  This bash script creates the submission tarball for GC: Salient360!
##
######################################################################
##
### Change Log:
##
##
######################################################################

folder="Salient360_xd_qsal"
[ -d ${folder} ] && rm -r ${folder}

mkdir ${folder}
cp *.m ${folder}
cp README.md ${folder}
pandoc -t html README.md -o ${folder}/README.html

tar czf ${folder}.tar.gz ${folder}

[ -d ${folder} ] && rm -r ${folder}
######################################################################
### create_submission.sh ends here
