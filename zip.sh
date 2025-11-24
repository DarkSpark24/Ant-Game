ARCHIVE_NAME="logic.zip"
rm ${ARCHIVE_NAME}
zip -r "${ARCHIVE_NAME}" main.py logic -x "logic/__pycache__/*"

echo "打包完成: ${ARCHIVE_NAME}"