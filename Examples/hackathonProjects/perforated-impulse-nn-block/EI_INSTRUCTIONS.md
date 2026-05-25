# Edge Impulse Instructions

When editing this project this file is to help get back up to speed with how it interfaces with Edge Impulse.

## Getting Datasets

To get datasets in the EI format go to the original NN block on the EI project and click to edit block locally.  When you unzip that download the np files will be in the data directory.

## Building the block
Build the block just with docker build.  Name can be whatever you like.

    docker build -t custom-ml-pytorch .

## Pushing the block
To push the block to Edge Impulse run the following command.  This is expected to take some time.  Once pushed you can simply reload your project, no need to delete and re-add the block.

    edge-impulse-blocks push