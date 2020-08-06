#!/bin/bash

#install the customized version of Reinier's reservoir package
chmod a+x ../reinstall.sh
../reinstall.sh

chmod a+x ./build_file_system.sh
./build_filesystem.sh

python execute.py

cd PyFiles; python test.py