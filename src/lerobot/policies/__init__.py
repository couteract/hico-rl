# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Policy configuration exports.

Some Evo-RL distributions intentionally ship only the policies used by the
benchmark. Keep optional policy imports lazy so the available policies (most
importantly SmolVLA and SAC) remain usable in that reduced distribution.
"""

try:
    from .act.configuration_act import ACTConfig as ACTConfig
except ModuleNotFoundError:
    ACTConfig = None
try:
    from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
except ModuleNotFoundError:
    DiffusionConfig = None
try:
    from .evo1.configuration_evo1 import Evo1Config as Evo1Config
except ModuleNotFoundError:
    Evo1Config = None
try:
    from .groot.configuration_groot import GrootConfig as GrootConfig
except ModuleNotFoundError:
    GrootConfig = None
try:
    from .pi0.configuration_pi0 import PI0Config as PI0Config
except ModuleNotFoundError:
    PI0Config = None
try:
    from .pi0_fast.configuration_pi0_fast import PI0FastConfig as PI0FastConfig
except ModuleNotFoundError:
    PI0FastConfig = None
try:
    from .pi05.configuration_pi05 import PI05Config as PI05Config
except ModuleNotFoundError:
    PI05Config = None
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
from .smolvla.processor_smolvla import SmolVLANewLineProcessor
try:
    from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig
except ModuleNotFoundError:
    TDMPCConfig = None
try:
    from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig
except ModuleNotFoundError:
    VQBeTConfig = None
try:
    from .wall_x.configuration_wall_x import WallXConfig as WallXConfig
except ModuleNotFoundError:
    WallXConfig = None
try:
    from .xvla.configuration_xvla import XVLAConfig as XVLAConfig
except ModuleNotFoundError:
    XVLAConfig = None
SARMConfig = None

__all__ = [
    "ACTConfig",
    "DiffusionConfig",
    "Evo1Config",
    "PI0Config",
    "PI05Config",
    "PI0FastConfig",
    "SmolVLAConfig",
    "SARMConfig",
    "TDMPCConfig",
    "VQBeTConfig",
    "GrootConfig",
    "XVLAConfig",
    "WallXConfig",
]
