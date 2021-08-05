import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back, Style

from traceback import extract_stack
from colors import H, W, FAIL

for x in extract_stack():
    if not x[0].startswith('<frozen importlib'):
        filename = x[0]
        break

def info(string):
    print(f"-- [INFO] -- {Fore.YELLOW}{filename}{Style.RESET_ALL} -- " + string)
    
def error(string):
    print(f"!! [{Fore.RED}ERROR{Style.RESET_ALL}] !! {Fore.YELLOW}{filename}{Style.RESET_ALL} -- " + string)


