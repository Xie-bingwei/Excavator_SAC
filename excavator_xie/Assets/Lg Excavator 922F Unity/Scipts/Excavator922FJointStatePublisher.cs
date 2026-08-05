using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;  // 【新增】引入这个，简化 TimeMsg 写法   
using System.Linq;
using AGXUnity; // 引入 AGX 的命名空间
using System;   // 用于 Serializable
using System.Reflection; // 【新增】引入反射命名空间，用于动态调用 Clock


// 【新增 1】定义一个配置类，让 Inspector 面板能显示 Offset 和 Multiplier
// [System.Serializable] 使得这个类可以在 Unity 编辑器里显示和折叠
[System.Serializable]
public class JointConfiguration
{
    public string rosJointName;      // (可选) URDF 里的关节名，不填则默认用物体名
    public Constraint agxConstraint; // 对应的 AGX 关节物体 (原 agxJoints 里的元素)
    public float offset = 0.0f;      // 【核心】零点偏移 (弧度)，用来修正姿态不一致
    public float multiplier = 1.0f;  // 方向乘数 (默认为1，因为你确认方向是对的)
}

public class Excavator922FJointStatePublisher : MonoBehaviour
{
    // Start is called before the first frame update
    // 配置部分
    [Header("ROS Settings")]
    public string topicName = "/unity/joint_states";
    public float publishRateHz = 50f; // 发布频率

    // 这里改成 AGX 的 Constraint 类型
    // 【修改 1】这里不再直接使用 Constraint 列表，而是使用我们自定义的配置列表
    // public List<Constraint> agxJoints;
    [Header("Joint Configuration")]
    public List<JointConfiguration> robotJoints; // (新代码)

    // === 【新增配置】时间源设置 ===
    [Header("Time Source")]
    [Tooltip("可选：拖入实现 Now(): TimeMsg 的 Clock 组件（如 ROS Clock）。留空则使用 Time.time。")]
    public MonoBehaviour clockSource;


    // ROS 连接器
    private ROSConnection ros;
    private float timeElapsed;

    // 【新增】用于缓存反射方法，优化性能
    private MethodInfo _clockMethod;
    private float _publishInterval; // 缓存发布间隔

    // ── RL 数据 (2026-08-04): 自动发现, 不改场景 ──
    private AGXUnity.RigidBody _baseFootprintRb;

    void Start()
    {
        // 获取 ROS 连接实例
        ros = ROSConnection.GetOrCreateInstance();
        // 注册发布者
        ros.RegisterPublisher<JointStateMsg>(topicName);

        // === 【新增】在启动时缓存 Clock 的 Now() 方法 ===
        if (clockSource != null)
        {
            _clockMethod = clockSource.GetType().GetMethod(
                "Now",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic
            );
        }

        // ── RL 数据: 自动发现, 不改场景 ──
        var baseGo = GameObject.Find("base_footprint");
        if (baseGo != null) _baseFootprintRb = baseGo.GetComponent<AGXUnity.RigidBody>();
    }

    // Update is called once per frame
    // void Update()
    // {
    //     timeElapsed += Time.deltaTime;

    //     if (timeElapsed > 1.0f / publishRateHz)
    //     {
    //         PublishJointStates();
    //         timeElapsed = 0;
    //     }        
    // }

    // 【修改 1】一定要用 FixedUpdate
    // AGX 的物理步进是在 FixedUpdate 发生的。
    // 用 Update 会导致你获取到的关节角度是“上一帧”的物理数据，但时间戳却是“当前帧”的渲染时间，导致滑步。
    void FixedUpdate()
    {
        timeElapsed += Time.fixedDeltaTime; // 使用 fixedDeltaTime

        if (timeElapsed >= _publishInterval)
        {
            PublishJointStates();
            timeElapsed = 0;
        }        
    }

    void PublishJointStates()
    {
        var msg = new JointStateMsg();

        // 设置 Header
        msg.header = new HeaderMsg();
        msg.header.frame_id = "base_footprint";
        // 或者 base_footprint，根据你的 TF 树根节点定
        // ROS2 时间戳处理比较麻烦，暂时留空或简单的计数，ROS2 会自动处理接收时间

        // === 【修改】使用封装好的时间策略获取时间戳 ===
        msg.header.stamp = GetSimTimeStamp();

        int jointCount = robotJoints.Count; 
        msg.name = new string[jointCount];
        msg.position = new double[jointCount];


        // // === 【重点修改区域：使用仿真时间】 ===

        // // 获取 Unity 当前仿真时间 (秒)
        // float currentTime = Time.time;

        // // 转换为 int 类型的秒 (ROS2 标准要求 int32) 
        // // 计算纳秒 (uint 类型)
        // int sec = (int)currentTime;
        // uint nanosec = (uint)((currentTime - sec) * 1e9);

        // // 赋值给 ROS 消息
        // msg.header.stamp = new TimeMsg(sec, nanosec);

        // === 【修改 2】循环逻辑更新 ===
        // int jointCount = agxJoints.Count;
        // int jointCount = robotJoints.Count; // 使用新的列表计数
        // msg.name = new string[jointCount];
        // msg.position = new double[jointCount];

        // 填充数据
        for (int i = 0; i < jointCount; i++)
        {
            var config = robotJoints[i];

            // 1. 名字处理
            // 如果你在 Inspector 里填了 rosJointName 就用它，没填就用物体名
            if (string.IsNullOrEmpty(config.rosJointName))
            {
                // 防止空引用报错
                if (config.agxConstraint != null)
                    msg.name[i] = config.agxConstraint.gameObject.name;
                else
                    msg.name[i] = "unknown_joint";
            }
            else
            {
                msg.name[i] = config.rosJointName;
            }

            // 2. 角度计算 (包含 Offset 修正)
            if (config.agxConstraint != null)
            {
                // 获取 AGX 原始角度
                double rawAngle = config.agxConstraint.GetCurrentAngle();

                // // =================【校准关键代码】Start =================
                
                // // 1. 打印原始角度到控制台
                // // 格式：[校准] 关节名 : 角度数值
                // Debug.Log($"[校准] Joint: {msg.name[i]} | AGX Raw Angle: {rawAngle}");

                // // 2. 暂时屏蔽 Offset
                // // 我们只发送 (raw * multiplier)，相当于强制 offset = 0
                // // 这样你看到的就是最原始的 AGX 行为
                // msg.position[i] = (rawAngle * config.multiplier); // + config.offset; <--- 已注释掉
                
                // // =================【校准关键代码】End ===================
                // Debug.Log($"[校准读数] {msg.name[i]} 当前角度: {rawAngle}");



                // 【核心修改】应用公式： (原始角度 * 1.0) + 偏移量
                // 你主要在 Inspector 里调整 offset 这个值
                msg.position[i] = (rawAngle * config.multiplier) + config.offset;
            }

            // // 1. 名字：使用 GameObject 的名字作为 Joint Name
            // msg.name[i] = agxJoints[i].gameObject.name;

            // // 2. 角度：从 AGX Constraint 获取当前角度
            // // GetCurrentAngle() 通常返回弧度，正好符合 ROS 标准
            // // 如果是 Prismatic (伸缩关节)，可能需要用 GetCurrentPosition()，这里默认按 Hinge (旋转) 处理
            // msg.position[i] = agxJoints[i].GetCurrentAngle();
        }

        // ── RL 数据: base angular velocity → velocity[1..3] ──
        msg.velocity = new double[jointCount];
        msg.effort   = new double[jointCount];
        if (_baseFootprintRb != null && _baseFootprintRb.Native != null)
        {
            try
            {
                var av = _baseFootprintRb.AngularVelocity;
                msg.velocity[1] = (double)av.x;
                msg.velocity[2] = (double)av.y;
                msg.velocity[3] = (double)av.z;
            }
            catch (System.Exception) {}
        }

        ros.Publish(topicName, msg);
    }

    // === 【新增】核心时间获取逻辑 ===
    // 优先尝试从 clockSource 获取 ROS 同步时间，失败则回退到 Unity 时间
    private TimeMsg GetSimTimeStamp()
    {
        // 策略 A: 如果绑定了外部时钟（如 ROSClock），尝试调用它的 Now() 方法
        if (clockSource != null && _clockMethod != null)
        {
            try
            {
                // 反射调用，获取 TimeMsg
                return (TimeMsg)_clockMethod.Invoke(clockSource, null);
            }
            catch 
            { 
                // 如果调用失败，静默失败并执行下方的回退逻辑
            }
        }

        // 策略 B: 回退使用 Unity 的 Time.time
        float t = Time.time;
        uint sec_u = (uint)t;
        uint nsec = (uint)((t - sec_u) * 1e9f);
        return new TimeMsg((int)sec_u, nsec);
    }
}
