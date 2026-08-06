"""Single-process Tron2 deployment for a train-time prefix-RTC checkpoint.

The GPU model, inference runner, keyboard state machine, and WebSocket robot
adapter run in one process. No ZeroMQ inference server or SSH tunnel is used.
The inherited real-action setting remains enabled; use
``--cfg-options inference.dry_run=True`` for a no-control model smoke test.
"""

_base_ = './pi05_paligemma_tron2_lora_rtc_finetune.py'

inference = dict(
    type='Tron2RTCInferenceRunner',
    remote_inference=None,
    action_chunk=50,
    async_execution=True,
    execute_horizon=None,
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=None,
    ),
    # Stable global IDs: flower first, task2 prompts in dataset task_index
    # order, and the separately trained fold-clothes task last.
    task_descriptions={
        '1':
        'Put the flowers in the vase',
        '2':
        'Put both dolls into the pink basket',
        '3':
        'Put both dolls into the gray basket',
        '4':
        'Put both pens into the pink basket',
        '5':
        'Put both pens into the gray basket',
        '6': ('Put a doll into the gray basket, and put the other doll into '
              'the pink basket.'),
        '7': ('Put a pen into the gray basket, and put the other  pen into '
              'the pink basket.'),
        '8': ('Put both dolls into the pink basket, and put both pens into '
              'the gray basket.'),
        '9': ('Put both dolls into the gray basket, and put both pens into '
              'the pink basket.'),
        '10':
        'Put all the objects into the pink basket.',
        '11':
        'Put all the objects into the gray basket.',
        '12':
        'fold clothes',
    },
)
