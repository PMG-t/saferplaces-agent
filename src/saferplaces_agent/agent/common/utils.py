# DOC: Generic utils

import os
import sys
import re
import ast
import uuid
import math
import hashlib
import datetime
import tempfile
import textwrap


from typing import Sequence

from langchain_openai import ChatOpenAI

from langchain_core.messages import RemoveMessage, AIMessage, ToolMessage, ToolCall




# REGION: [Generic utils]

_temp_dir = os.path.join(tempfile.gettempdir(), 'saferplaces-agent')
os.makedirs(_temp_dir, exist_ok=True)


def guid():
    return str(uuid.uuid4())

def hash_string(s, hash_method=hashlib.md5):
    return hash_method(s.encode('utf-8')).hexdigest()

def s3uri_to_https(s3_uri):
    return s3_uri.replace('s3://', 'https://s3.us-east-1.amazonaws.com/')


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
            
            # DOC: LLM can asnwer with a python code block, so we need to extract the code and evaluate it
            if type(content) is str and content.startswith('```python'):
                content = content.split('```python')[1].split('```')[0]
            if type(content) is str and content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0]
            
            # DOC: LLM can answer with a python dict but sometimes as json, so we need to convert some values from json to py
            content = re.sub(r'\bnull\b', 'None', content) # DOC: replace null with None
            content = re.sub(r'\btrue\b', 'True', content) # DOC: replace true with True
            content = re.sub(r'\bfalse\b', 'False', content) # DOC: replace false with False
            
            return ast.literal_eval(content)
        except: 
            pass
    return llm_out.content


def map_action_new_layer(layer_name, layer_src, layer_styles=[]):
    """Create a map action with the given type and data."""
    layer_styles = { 'styles': layer_styles } if len(layer_styles) > 0 else dict()
    action_new_layer = {
        'action': 'new_layer',
        'layer_data': {
            'name': layer_name,
            'type': 'vector' if layer_src.endswith('.geojson') else 'raster',   # TODO: add more extensions (e.g. .gpkg, .tif, tiff, geotiff, etc.)
            'src': s3uri_to_https(layer_src),
            ** layer_styles
        }
    }
    return action_new_layer

# ENDREGION: [LLM and Tools]



# REGION: [Message utils funtion]

def merge_sequences(left: Sequence[str], right: Sequence[str]) -> Sequence[str]:
    """Add two lists together."""
    return left + right

def merge_dictionaries(left: dict, right: dict) -> dict:
    """Add two dictionaries together but merging ad all levels."""
    for key, value in right.items():
        if key in left:
            if isinstance(left[key], dict) and isinstance(value, dict) and len(value) > 0:
                left[key] = merge_dictionaries(left[key], value)
            elif isinstance(left[key], list) and isinstance(value, list):
                left[key] = left[key] + value
            else:
                left[key] = value
        else:
            left[key] = value
    return left            
            

def is_human_message(message):
    """Check if the message is a human message."""
    return hasattr(message, 'role') and message.role == 'human'


def remove_message(message_id):
    return RemoveMessage(id = message_id)

def remove_tool_messages(tool_messages):
    if type(tool_messages) is not list:
        return remove_message(tool_messages.id)
    else:
        return [remove_message(tm.id) for tm in tool_messages]
    
    
def build_tool_call_message(tool_name, tool_args=None, tool_call_id=None, message_id=None, message_content=None):
    message_id = hash_string(datetime.datetime.now().isoformat()) if message_id is None else message_id
    message_content = "" if message_content is None else message_content
    tool_call_id = hash_string(message_id) if tool_call_id is None else tool_call_id
    tool_call_message = AIMessage(
        id = message_id,
        content = message_content,
        tool_calls = [
            ToolCall(
                id = tool_call_id,
                name = tool_name,
                args = tool_args if tool_args is not None else dict()
            )
        ]
    )
    return tool_call_message

# ENDREGION: [Message utils funtion]