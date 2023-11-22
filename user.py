from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='perform user tasks')
parser.add_argument('-t', '--task', type=str, default=None, help='which task to perform')
parser.add_argument('--noupgrade', default=False, action='store_true')
args = parser.parse_args()

# This is just to store some locally used variables not relevant for any other user of roicat
# Right now I suppose it's just to connect myself to other code I've written.
def codePath():
    return Path('C:/Users/andrew/Documents/GitHub/vrAnalysis')

def remotePath():
    return "https://github.com/landoskape/vrAnalysis"
    
def localDataPath():
    return Path("C:/Users/andrew/Documents/localData")

def analysisPath():
    return localDataPath() / 'analysis'

def updateVR(upgrade=True):
    # convenience function that outputs the pip command for updating vrAnalysis
    extra_args = "--force-reinstall --no-deps " if upgrade else ""
    cmdPromptCommand = "pip install " + extra_args + f"git+{remotePath()}"
    print(cmdPromptCommand)

if args.task=='updateVR':
    updateVR(upgrade=not(args.noupgrade))
