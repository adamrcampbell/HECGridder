
HEC_ROOT=$PWD

# Remove build
rm -rf build

# Perform clean build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

# Execute 
./gridder

# Perform FFT and Convolution Correction
HEC_IMG_REAL=$HEC_ROOT/data/hec_grid_real.csv
HEC_IMG_IMAG=$HEC_ROOT/data/hec_grid_imag.csv
HEC_IMG_DIRTY=$HEC_ROOT/data/hec_dirty_right.csv
~/Desktop/HPC/Projects/FFT2DForGridder/Debug/FFT2DForGridder $HEC_IMG_REAL $HEC_IMG_IMAG $HEC_IMG_DIRTY

# Run Python Viewer
cd $HEC_ROOT/data
python3 viewer.py