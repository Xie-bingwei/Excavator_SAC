# msgs/ExcavatorState.msg - 完整状态向量(对接SAC)
std_msgs/Header header
float64[4]  q                     # 关节角度
float64[4]  q_dot                 # 关节角速度
float64[3]  p_tip                 # 齿尖位置
float64[3]  tracking_error        # 跟踪误差
float64     s_star                # 轨迹进度
float64     d_pipe                # 到管线距离
float64     alpha_margin          # 稳定裕度
float64     d_trench              # 到沟槽距离
float64     m_load_est            # GMO 负载估计(暂为 0)