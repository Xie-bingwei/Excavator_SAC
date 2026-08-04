using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
// using RosMessageTypes.Control; // 【关键】引用 Control 消息命名空间
// 如果报错找不到，请先在 Robotics -> Generate ROS Messages 里生成 control_msgs

/// <summary>
/// 监听 ROS 关节指令 (JointState)，控制挖掘机的工作装置。
/// </summary>
public class RosWorkDeviceSubscriber : MonoBehaviour
{
    [Header("Controller Reference")]
    // 引用你的挖掘机 AGX 控制器
    public Excavator922FTrackController excavatorController;

    [Header("ROS Topic Settings")]
    // MoveIt 或外部节点发布的控制话题
    public string topicName = "/unity/joint_command";
    
    [Header("URDF Joint Names")]
    // 必须与 model_gazebo.xacro 文件中的 joint name 完全一致
    public string swingJointName = "base_to_body_joint";
    public string boomJointName = "body_to_boom_joint";
    public string armJointName = "boom_to_arm_joint";
    public string bucketJointName = "arm_to_bucket_joint";
    
    // 【新增】安全锁：记录 ROS 是否已经发送了合理的非零初始指令
    private bool rosHasSynced = false;

    void Start()
    {
        var ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<JointStateMsg>(topicName, OnJointCmdReceived);
        // ros.Subscribe<JointTrajectoryControllerStateMsg>(topicName, OnJointCmdReceived);

    }

    void OnJointCmdReceived(JointStateMsg msg)
    {
        if (excavatorController == null) return;

        // 这里的 msg.joint_names 存储了关节名字
        // msg.reference.positions 存储了 MoveIt 规划出的瞬时目标位置
        // 注意：reference 是 TrajectoryPointMsg 类型

        // 获取 ROS 发来的关节名列表
        var jointNames = msg.name;
        // 获取 ROS 发来的目标位置 (其实就是目标角度)
        var targetPositions = msg.position;

        // ==========================================
        // 【核心】防跳变拦截器：拿 Boom 的数据做安检
        // ==========================================
        float cmdBoom = 0f;
        bool foundBoom = false;

        // 先找到 ROS 发来的 Boom 指令是多少
        for (int i = 0; i < jointNames.Length; i++)
        {
            if (jointNames[i] == boomJointName && i < targetPositions.Length)
            {
                cmdBoom = (float)targetPositions[i];
                foundBoom = true;
                break;
            }
        }

        // 如果还没同步，检查指令是否合理
        if (!rosHasSynced && foundBoom)
        {
            // 期待 ROS 发回来的 Boom 应该在 0.765 附近。
            // 如果发来的是 0.0，两者的差值绝对值就是 0.765。我们设一个阈值 0.5。
            if (Mathf.Abs(cmdBoom - 0.765f) > 0.5f)
            {
                Debug.Log($"[安全锁生效] 收到异常初始指令 Boom={cmdBoom}，这是 ROS 没睡醒发的，直接丢弃！");
                return; // 直接退出，绝不执行！
            }
            else
            {
                Debug.Log($"[同步成功] 收到合理指令 Boom={cmdBoom}，安全锁已解开！");
                rosHasSynced = true; // 解锁
            }
        }

        // 没解锁之前，绝不往下执行任何关节操作
        if (!rosHasSynced) return;
        // ==========================================

        // 遍历消息里所有的关节，找到我们关心的那几个
        for (int i = 0; i < jointNames.Length; i++)
        {
            string name = jointNames[i];

            // 确保有对应的位置数据
            if (i >= targetPositions.Length) continue;

            float pos = (float)targetPositions[i]; // 目标角度 (弧度)

            // 根据名字匹配，调用控制器的设置函数
            // offset 与发布端 (JointStatePublisher) 一致: ROS = AGX_raw + offset → AGX_raw = ROS - offset
            if (name == swingJointName)
                excavatorController.SetRosSwing(pos - 0.0f);
            else if (name == boomJointName)
                excavatorController.SetRosBoom(pos - 0.765f);
            else if (name == armJointName)
                excavatorController.SetRosArm(pos - (-0.743f));
            else if (name == bucketJointName)
                excavatorController.SetRosBucket(pos - (-0.05f));
        }
    }    
}
