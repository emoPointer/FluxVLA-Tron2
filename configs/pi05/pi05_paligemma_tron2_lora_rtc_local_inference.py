"""Native tron2_env deployment for the fold-clothes prefix-RTC checkpoint.

The FluxVLA model remains local, but observation and robot execution are owned
by the pinned public ``tron2_env.Tron2Env`` implementation.  RTC timing follows
the public ``tron2_openpi`` trained-RTC client profile.  There is no ZeroMQ
policy server or SSH tunnel in this adapter.
"""

_base_ = './pi05_paligemma_tron2_lora_rtc_finetune.py'

# Match the selected fold-clothes 18k checkpoint's saved train-time RTC
# configuration. These values come from its checkpoint-local config.json.
_fold_rtc_training_config = dict(
    enabled=True,
    max_delay=20,
    distribution='uniform',
    delay_values=[0, 5, 10, 19],
    temperature=1.0,
)

model = dict(rtc_training_config=_fold_rtc_training_config)
inference_model = dict(rtc_training_config=_fold_rtc_training_config)

inference = dict(
    type='Tron2RTCInferenceRunner',
    remote_inference=None,
    action_chunk=50,
    async_execution=False,
    execute_horizon=None,
    mixed_precision_dtype='bf16',
    enable_mixed_precision=True,
    # Training proprio is 18-D (16 deployed joints/grippers + measured head),
    # while robot commands remain the 16-D action layout.
    include_head_in_state=True,
    rtc_config=dict(
        enabled=True,
        method='prefix',
        # Public tron2_openpi trained-RTC client defaults.
        delay=6,
        execution_horizon=10,
        trigger_poll_interval_s=0.005,
        observation_timeout_budget_s=5.0,
        recovery_blend_frames=6,
        # Keep the model condition fixed at a checkpoint-supported prefix.
        # ActionQueue still crops by the actual number of consumed frames.
        prefix_len=19,
        # Training supervises only the 16 deployed action dimensions. Feed
        # those dimensions back, append the measured normalized head, and let
        # PI0.5 zero-pad the remaining tail to its 32-D width.
        prefix_action_dim=16,
        prefix_head_from_observation=True,
        action_postprocess=dict(
            enabled=False,
            boundary_blend_frames=0,
            boundary_blend_curve='smoothstep',
            boundary_blend_scope='arm',
            ema_alpha=1.0,
            ema_frames=0,
            ema_scope='arm',
        ),
    ),
    operator=dict(
        _delete_=True,
        type='Tron2NativeEnvOperator',
        # All observations and commands use the public WebSocket runtime.
        # Images come from Bridge; state comes from the robot WebSocket, which
        # matches the public trained-RTC deployment profile.
        bridge_host='wss://10.192.1.4',
        bridge_ws_path='/bridge/ws',
        bridge_image_topics=dict(
            camera_left='/camera/left/color/image_resized/compressed',
            camera_right='/camera/right/color/image_resized/compressed',
            camera_top='/camera/top/color/image_raw/compressed',
        ),
        bridge_joint_topics=dict(
            joint_states='/joint_states',
            gripper='/gripper_state',
        ),
        bridge_image_max_fps=0,
        bridge_align_max_delay_ms=200,
        bridge_verify_tls=False,
        bridge_state_source='legacy',
        robot_ip='10.192.1.2',
        ws_port=5000,
        state_dim=18,
        fps=30.0,
        publish_rate=300.0,
        state_queue_maxlen=7,
        state_polling_rate=200.0,
        connection_timeout=5.0,
        # Keep the established task-ID-0 arm target while delegating the
        # actual MoveJ bring-up route to tron2_env. Head is never commanded.
        init_joints=[
            0.0,
            0.24,
            0.0,
            -1.56,
            0.24,
            0.0,
            0.0,
            0.0,
            -0.24,
            0.0,
            -1.56,
            -0.24,
            0.0,
            0.0,
        ],
        init_head=None,
        init_ee_z_min=-0.6,
        init_gripper_opening=1.0,
        # Idle r: open both grippers and allow them to settle before MoveJ.
        reset_gripper_open_wait_s=0.5,
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
