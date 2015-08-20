# vision

this is ... we'll see

# installing on Mac OS 10.10.5

    sudo port selfupdate
    sudo port install python27 py27-numpy py27-scipy
    sudo port install opencv +python27
    sudo port select --set python python27

# getting data

## labeled faces in the wild (lfw)

Available here: http://vis-www.cs.umass.edu/lfw/lfw.tgz

    cd data
    curl http://vis-www.cs.umass.edu/lfw/lfw.tgz > lfw.tgz
    tar zxvf lfw.tgz
    rm lwf.tgz
    






