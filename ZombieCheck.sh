#!/bin/bash

SEARCH_FOLDER=$1

# Loop over 5 folders
for dir1 in $SEARCH_FOLDER
do
    if [ -d "$dir1" ]
    then
        for dir2 in $dir1/*
        do
            if [ -d "$dir2" ]
            then
                for dir3 in $dir2/*
                do
                    if [ -d "$dir3" ]
                    then
                        # check that file has not been corrupted
                        echo "checking folder " ${dir3}/
                        ./ZombieHunting ${dir3}/
                    fi
                done
            fi
        done
    fi
done
