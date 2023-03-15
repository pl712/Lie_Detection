import subprocess

command = '''#!/usr/bin bash
cd /home/ubuntu/videos
for fileName in ./*; do
    echo $fileName
    /home/ubuntu/OpenFace/build/bin/FeatureExtraction -f $fileName -gaze -aus
done
'''

subprocess.run(command, shell=True)