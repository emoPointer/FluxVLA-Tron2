"""PI0.5 Tron2 LoRA fine-tuning with training-time RTC conditioning.

This inherits the established rank-256 Tron2 LoRA recipe and changes only the
training objective: each sample receives an exponentially distributed known
action prefix of zero to nine steps. Prefix actions are clean conditions and
are excluded from the flow-matching loss. The 50-step action horizon, LoRA
targets, optimizer, normalization, and checkpoint policy remain inherited.
"""

_base_ = './pi05_paligemma_tron2_lora_finetune.py'

model = dict(
    rtc_training_config=dict(
        enabled=True,
        max_delay=10,
        distribution='exponential',
        temperature=1.0,
    ), )

# Keep construction of the optional inference model aligned with the training
# model. Deployment uses the separate local-inference config so this training
# recipe does not silently change hardware execution behavior.
inference_model = dict(
    rtc_training_config=dict(
        enabled=True,
        max_delay=10,
        distribution='exponential',
        temperature=1.0,
    ), )
