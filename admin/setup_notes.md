# Feather
To install feather, I had to use Linux (the version on Google Cloud) and the following commmand:

  conda install feather-format -c conda-forge

In Python, the line is "import feather" (not feather-format)

To run script 003 and unzip the files, combine them, and save as feather, I had to up my RAM on the instance to 6.5 GB. Even then, if I had any other RAM being used it would sometimes hit the limit, so 8 is probably a safer bet. 
