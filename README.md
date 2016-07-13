# ProjetL2

Developped and tested on Windows 10 with [WinPython 3.4.4.1](https://winpython.github.io/) which is a Python 3.4 implementation.

The data used for testing (thus, the formatting supported for pulling data) was taken from [here](http://www-connex.lip6.fr/~baskiotisn/pldac//addic7ed.tgz).

In addition to the libraries provided by WinPython, the followings were also used:  
-[treetaggerwrapper](https://perso.limsi.fr/pointal/dev:treetaggerwrapper)  
-[TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)(not a library)

The code is organised between 3 files(Grapher,Series,My_lil_matrix), My_lil_matrix doesn't call Series which doesn't call Grapher.  
This makes My_lil_matrix independant from the others and easier to reuse.

In addition, Test.py is used to keep all the common test routines in a single file. It should be used as a main file for ease of use but it doesn't make additionnal import calls to Grapher.py, feel free to edit it for testing purposes.

The simplest way to use this after setting the paths would be calling Test.g() then Test.TestTotal().

Things to note : 
For ease of use, Series.py sets a few paths for my machine but you need to comment the current ones and add your own paths for it to work.  
pathProj is only used once, and only needs to be set if Test.names() is called in which case you need to have a names.txt file. Can be disregarded most of the time. 
pathDumps should be an empty folder if used for the first time, it will contain folders to organize data and images.  
pathData should contain the data downloaded and extracted from [here](http://www-connex.lip6.fr/~baskiotisn/pldac//addic7ed.tgz).(this should have folders named after series)

Pulling the data from the files takes some time so it's advised to start with small number of series.
Any feedback is welcome!
