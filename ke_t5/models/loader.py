# Copyright 2021 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, Type
import importlib


MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str) -> Callable[[Type], Type]:
    """
    Register an model to be available in command line calls.

    >>> @register_model("my_model")
    ... class My_Model:
    ...     pass
    """

    def _inner(cls_):
        global MODEL_REGISTRY
        MODEL_REGISTRY[name] = cls_
        return cls_

    return _inner


def _camel_case(name: str):
    words = name.split('_')
    class_name = ''
    for w in words:
        class_name += w[0].upper() + w[1:]
    return class_name


def load_model(model_path: str):
    global MODEL_REGISTRY
    if model_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_path]
    
    if ':' in model_path:
        path_list = model_path.split(':')
        module_name = path_list[0]
        class_name = _camel_case(path_list[1])
    elif '/' in model_path:
        path_list = model_path.split(':')
        module_path = path_list[0].split('/')
        module_name = '.'.join(module_path)
        class_name = _camel_case(path_list[1])
    
    my_module = importlib.import_module(module_name)
    model_class = getattr(my_module, class_name)
    return model_class
