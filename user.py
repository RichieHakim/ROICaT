import sys
from pathlib import Path

assert len(sys.argv) <= 2, "user.py only accepts one input argument"
task = sys.argv[1] if len(sys.argv)==2 else None

# This is just to store some locally used variables not relevant for any other user of roicat
# Right now I suppose it's just to connect myself to other code I've written.
def codePath():
    return Path('C:/Users/andrew/Documents/GitHub/vrAnalysis')

def localDataPath():
    return Path("C:/Users/andrew/Documents/localData")

def analysisPath():
    return localDataPath() / 'analysis'

def updateVR():
    # convenience function that outputs the pip command for updating vrAnalysis
    cmdPromptCommand = f"pip install --upgrade {codePath()}"
    print(cmdPromptCommand)

if task=='updateVR': updateVR()
