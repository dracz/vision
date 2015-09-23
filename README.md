# vision

this is ... we'll see

# installing on Mac OS 10.10.5

    sudo port selfupdate
    sudo port install python27 py27-numpy py27-scipy py27-pil
    sudo port install opencv +python27
    sudo port select --set python python27

# getting data

Input data is expected to be unpacked in a path relative to the code:

    ../../img
    

## labeled faces in the wild (lfw)

Download here:
http://vis-www.cs.umass.edu/lfw/lfw.tgz

    cd data
    curl http://vis-www.cs.umass.edu/lfw/lfw.tgz > lfw.tgz
    tar zxvf lfw.tgz
    rm lwf.tgz
    
## labeled faces in the wild, cropped (lfwc)

Download here:
http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip









