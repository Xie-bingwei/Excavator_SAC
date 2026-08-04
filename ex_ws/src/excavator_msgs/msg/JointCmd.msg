# msgs/JointCmd.msg - APF输出的关节指令
float64[4] target_position # 目标关节角度 [swing, boom, arm, bucket](rad)
float64[4] target_speed # 目标关节速度(rad/s)
float64[2] track_speed # 左右履带速度 [left, right] (rad/s)
uint8 control_mode # 0=位置, 1=速度, 2=力矩