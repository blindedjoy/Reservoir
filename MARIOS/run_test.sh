#!/bin/bash

#estimated number of cores:

# 16 tests, 8 cores each. Then we have the cv loop, requesting four cores per run.
# 16 * 8

#install the customized version of Reinier's reservoir package
cd ..
chmod a+x ./reinstall.sh
./reinstall.sh
cd MARIOS

chmod a+x ./build_file_system.sh
./build_filesystem.sh

chmod a+x ./execute_test.py
python execute.py

#python PyFiles/test.py