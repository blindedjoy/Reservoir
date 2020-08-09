#!/bin/bash

#install the customized version of Reinier's reservoir package
cd ..
chmod a+x ./reinstall.sh
./reinstall.sh
cd MARIOS

chmod a+x ./build_file_system.sh
./build_filesystem.sh

chmod a+x ./execute.py
python execute.py

#python PyFiles/test.py