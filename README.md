# ProjetL2

#### Setup

Developed and tested on Windows 10 with [WinPython 3.4.4.1](https://winpython.github.io/), which is a Python 3.4 implementation.

In addition to the libraries provided by WinPython, the followings were also used:  
* [treetaggerwrapper](https://perso.limsi.fr/pointal/dev:treetaggerwrapper)  
* [TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)(not a library)

The data used for testing  - and thus the formatting supported for input data - was taken from [here](http://www-connex.lip6.fr/~baskiotisn/pldac//addic7ed.tgz). Warning: This is a 1.3GB file.

#### Usage

The code is organised between 3 Python files with the following dependencies:
```
Grapher.py -> Series.py -> My_lil_matrix.py
```

In addition, `Test.py` is used to keep all the common test routines in a single file. It should be used as a main file for ease of use, but it doesn't make additional import calls to `Grapher.py`. Feel free to edit it for testing purposes.

After setting the environment variables, the simplest way to use this would be to call `go('Test')` then `TestTotal('Test')`.

This is best used in an interactive console.

Notes:
* For ease of use, `Series.py` sets a few paths in the environment. You might need to edit the default paths to match your own paths for it to work.  
* `pathProj` is only used once, and only needs to be set if `Test.names()` is called in which case you need to have a `names.txt` file. Can be disregarded most of the time.
* `pathDumps` should be an empty folder if used for the first time, it will contain folders to organize data and images.  
* `pathData` should contain the data downloaded and extracted beforehand. (This should have folders named after series.) The path to a subtitle .txt file looks like this  `pathData + '/1245___Game_of_Thrones/01/01__Winter_is_Coming.txt' ` .
* Pulling the data from the files takes some time so it's advised to start with a small number of series.

Any feedback is welcome!
