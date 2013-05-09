#!/bin/bash
# Bash script to run test codes (CUDA Day 2)

# Written for timing CUDA codes in SHARCNET Summer School hands on session.

if test $# -ne 1
  then
  echo "Script to time CUDA runs"
  echo
  echo "Arguments:  executable_name"
  echo
  exit
  fi

echo
echo  Individual timings:

\rm __results &> /dev/null
$1 | grep "^Time: " | cut -d" " -f2 | tee -a __results

echo
echo  The best timing, average, and std:
cat __results | awk '{x=$1; sum+=x; sum2+=x*x; N+=1; if(min==""){min=$1}; if($1< min) {min=$1}} END {printf "%e %e %e\n", min, sum/N, sqrt(sum2/N-sum*sum/N/N)}'
echo

\rm __results &> /dev/null

