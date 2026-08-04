# msgs/APFDebug.msg - APF调试信息
std_msgs/Header header
float64[3] f_att # 引力向量
float64[3] f_rep_pipe # 管线斥力向量
float64[4] tau_rep_tip # 倾覆斥力力矩
float64[2] f_rep_trench # 沟槽斥力(底盘 2D)
float64 s_star # 轨迹最近点参数
float64 tracking_err # 跟踪误差范数
float64[4] tau_cmd # 最终关节力矩指令
string active_phase # 当前挖掘阶段名称