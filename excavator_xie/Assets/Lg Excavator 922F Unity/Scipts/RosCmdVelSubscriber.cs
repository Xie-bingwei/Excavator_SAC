using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RosCmdVelSubscriber : MonoBehaviour
{
    [Header("AGX Controller Reference")]
    // 引用你的 AGX 控制器脚本
    public Excavator922FTrackController excavatorController;
    
    [Header("ROS Settings")]
    public string cmdVelTopic = "/cmd_vel_nav";

    [Header("AGX Physics Params")]
    [Tooltip("左右履带中心距 (米),用于差速计算。922F 约 2.6m")]
    public float trackWidth = 2.231f;

    [Tooltip("驱动轮(Sprocket)半径 (米)。用于将 m/s 换算为 AGX 电机需要的 rad/s")]
    public float sprocketRadius = 0.25f;
    void Start()
    {
        
        var ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<TwistMsg>(cmdVelTopic, OnCmdVelReceived);
        
    }

    void OnCmdVelReceived(TwistMsg msg)
    {
        if (excavatorController == null) return;

        // 1. 获取 ROS 期望速度
        float v = (float)msg.linear.x; // 前进 (m/s)
        float w = (float)msg.angular.z; // 转向 (rad/s)

        // 2. 差速计算 (履带运动学)
        // 左履带线速度 = v - (w * width / 2)
        // 右履带线速度 = v + (w * width / 2)
        float v_left_m_s  = v - (w * trackWidth / 2.0f);
        float v_right_m_s = v + (w * trackWidth / 2.0f);

        // 3. 转换为 AGX Hinge 需要的角速度 (rad/s) = 线速度 / 半径
        float left_rad_s = v_left_m_s / sprocketRadius;
        float right_rad_s = v_right_m_s / sprocketRadius;

        // 4. 发送给控制器
        excavatorController.SetRosTrackVelocity(left_rad_s, right_rad_s);



    }

   
}
