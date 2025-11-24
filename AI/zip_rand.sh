mkdir -p build
cd build

ARCHIVE_NAME="ai_rand.zip"
SOURCE_SCRIPT="ai_random_safe.py"

ln -s ../ai_main.py main.py
ln -s "../${SOURCE_SCRIPT}" ai.py
ln -s ../../logic logic

zip -r ../"${ARCHIVE_NAME}" main.py logic ai.py -x "logic/__pycache__/*"

cd ..
rm -r build