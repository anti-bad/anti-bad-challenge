#!/bin/bash
# Create submission.zip from all files in submission folder

cd submission
zip -q -r ../submission.zip *
cd ..

echo "Created submission.zip"
ls -lh submission.zip
