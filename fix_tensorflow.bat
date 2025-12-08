@echo off
REM Fix TensorFlow Installation on Windows
echo Uninstalling existing TensorFlow...
pip uninstall tensorflow tensorflow-cpu tensorflow-gpu -y

echo.
echo Installing TensorFlow 2.15.0 (Windows compatible)...
pip install tensorflow-cpu==2.15.0

echo.
echo Testing TensorFlow installation...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('✓ TensorFlow working!')"

echo.
echo Done! Press any key to exit...
pause