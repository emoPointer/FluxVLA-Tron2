# Copyright 2026 Limx Dynamics
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

import os

if os.getenv('FLUXVLA_REMOTE_CLIENT_ONLY', '0') == '1':
    from .base_inference_runner import BaseInferenceRunner  # noqa: F401, F403
    from .tron2_inference_runner import \
        Tron2InferenceRunner  # noqa: F401, F403
else:
    from .aloha_inference_runner import \
        AlohaInferenceRunner  # noqa: F401, F403
    from .aloha_rtc_inference_runner import \
        AlohaRTCInferenceRunner  # noqa: F401, F403
    from .base_train_runner import BaseTrainRunner  # noqa: F401, F403
    from .ddp_train_runner import DDPTrainRunner  # noqa: F401, F403
    from .fsdp_train_runner import FSDPTrainRunner  # noqa: F401, F403
    from .libero_eval_runner import LiberoEvalRunner  # noqa: F401, F403
    from .libero_inference_runner import \
        LiberoInferenceRunner  # noqa: F401, F403
    from .tron2_inference_runner import \
        Tron2InferenceRunner  # noqa: F401, F403
    from .tron2_rtc_inference_runner import \
        Tron2RTCInferenceRunner  # noqa: F401, F403
    from .ur_inference_runner import URInferenceRunner  # noqa: F401, F403
