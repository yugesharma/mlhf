#! /bin/bash

PORT=22001
MACHINE=paffenroth-23.dyn.wpi.edu
KEY=$HOME/Documents/cs553/caseStudy1/keys/my_key
ssh -i $KEY -p ${PORT} -o BatchMode=yes -o StrictHostKeyChecking=no student-admin@${MACHINE}
