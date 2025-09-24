#! /bin/bash

PORT=22001
MACHINE=paffenroth-23.dyn.wpi.edu
KEY=$HOME/Documents/CS553/caseStudy2/keys/my_key
ssh -i $KEY -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE}
