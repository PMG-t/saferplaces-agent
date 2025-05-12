# DOC: Generic utils

import os
import sys
import re
import ast
import uuid
import math
import tempfile
import textwrap


from typing import Sequence

from langchain_openai import ChatOpenAI

from langchain_core.messages import RemoveMessage, AIMessage




# REGION: [Generic utils]

_temp_dir = os.path.join(tempfile.gettempdir(), 'saferplaces-agent')
os.makedirs(_temp_dir, exist_ok=True)


def guid():
    return str(uuid.uuid4())


def python_path():
    """ python_path - returns the path to the Python executable """
    pathname, _ = os.path.split(normpath(sys.executable))
    return pathname

def normpath(pathname):
    """ normpath - normalizes the path to use forward slashes """
    if not pathname:
        return ""
    return os.path.normpath(pathname.replace("\\", "/")).replace("\\", "/")

def juststem(pathname):
    """ juststem - returns the file name without the extension """
    pathname = os.path.basename(pathname)
    root, _ = os.path.splitext(pathname)
    return root

def justpath(pathname, n=1):
    """ justpath - returns the path without the last n components """
    for _ in range(n):
        pathname, _ = os.path.split(normpath(pathname))
    if pathname == "":
        return "."
    return normpath(pathname)

def justfname(pathname):
    """ justfname - returns the basename """
    return normpath(os.path.basename(normpath(pathname)))

def justext(pathname):
    """ justext - returns the file extension without the dot """
    pathname = os.path.basename(normpath(pathname))
    _, ext = os.path.splitext(pathname)
    return ext.lstrip(".")

def forceext(pathname, newext):
    """ forceext - replaces the file extension with newext """
    root, _ = os.path.splitext(normpath(pathname))
    pathname = root + ("." + newext if len(newext.strip()) > 0 else "")
    return normpath(pathname)


def try_default(f, default_value=None):
    """ try_default - returns the value if it is not None, otherwise returns default_value """
    try:
        value = f()
        return value
    except Exception as e:
        return default_value
    
     
def floor_decimals(number, decimals=0):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def ceil_decimals(number, decimals=0):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def dedent(s: str, add_tab: int = 0, tab_first: bool = True) -> str:
    """Dedent a string by removing common leading whitespace."""
    out = textwrap.dedent(s).strip()
    if add_tab > 0:
        out_lines = out.split('\n')
        tab = ' ' * 4
        out = '\n'.join([tab * add_tab + line if (il==0 and tab_first) or (il>0) else line for il,line in enumerate(out_lines)])
    return out

    
# ENDREGION: [Generic utils]



# REGION: [LLM and Tools]

_base_llm = ChatOpenAI(model="gpt-4o-mini")

def ask_llm(role, message, llm=_base_llm, eval_output=False):
    llm_out = llm.invoke([{"role": role, "content": message}])
    if eval_output:
        try: 
            content = llm_out.content
            print('\n\n')
            print(type(content))
            print(content)
            print('\n\n')
            if type(content) is str and content.startswith('```python'):
                content = content.split('```python')[1].split('```')[0]
            return ast.literal_eval(content)
        except: 
            pass
    return llm_out.content

# ENDREGION: [LLM and Tools]



# REGION: [Message utils funtion]

def merge_sequences(left: Sequence[str], right: Sequence[str]) -> Sequence[str]:
    """Add two lists together."""
    return left + right

def remove_message(message_id):
    return RemoveMessage(id = message_id)

def remove_tool_messages(tool_messages):
    if type(tool_messages) is not list:
        return remove_message(tool_messages.id)
    else:
        return [remove_message(tm.id) for tm in tool_messages]
    
# ENDREGION: [Message utils funtion]