#! /bin/bash
cd capstone/keras-frcnn-master
rm -r outgoing_images
mkdir outgoing_images 
python test_frcnn.py -p ./incoming_images
exit
exit