using System;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Nav;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using RosMessageTypes.BuiltinInterfaces;
using AGXUnity;

/// <summary>
/// 挖掘机专用 Odom/RTK 发布脚本 (AGX 适配完整版)
/// 适配文件名: Excavator922FOdomPublisher
/// 功能： 
/// 1. Odom: 基于 base_footprint (底盘) 发布里程计，速度来自 AGX 物理引擎。
/// 2. RTK: 基于驾驶室顶部的天线发布 GPS 坐标 (自动跟随驾驶室旋转)。
/// 挖掘机专用 Odom 发布脚本 (微分速度版 + 安全防护)
/// 功能：
/// 1. 解决 "Unity跑得快，RViz追不上" 的积分滞后问题。
/// 2. 解决 "车身侧躺" 和 "向左滑" 的坐标系问题。
/// 3. 内置防瞬移、防抖动算法。
/// 挖掘机专用 Odom 发布脚本 (终极修正版)
/// 1. 【速度同步】弃用物理引擎速度，改用 (位置差/时间差) 手动计算，彻底解决 RViz 追不上 Unity 的问题。
/// 2. 【坐标修正】修复了 "向前走变向左滑" (映射 -X 为前方) 和 "车身侧躺" (静态旋转补偿) 的问题。
/// 3. 【安全防护】内置防瞬移 (MaxSpeed) 和防抖动 (Deadzone) 保护。
/// </summary>
public class Excavator922FOdomPublisher : MonoBehaviour
{
    [Header("1. Odom Target (Base Footprint)")]
    [Tooltip("【重要】请拖入挖掘机的 'base_footprint' 或 'base_link' (履带/底盘部分)。\n这是 Odom 追踪的目标。")]
    public Transform baseFootprint;

    [Tooltip("这里留着仅作引用，本脚本不再读取它的物理速度")]
    public AGXUnity.RigidBody trackRigidbody;

    [Tooltip("里程计话题名")]
    public string odomTopic = "/bynav/odom";
    [Tooltip("TF Frame ID: 里程计坐标系")]
    public string odomFrameId = "odom";
    [Tooltip("Child Frame ID: 机器人底盘坐标系")]
    public string baseFrameId = "base_footprint";

    [Header("2. RTK Antennas (Cab/Swing)")]
    [Tooltip("左侧 RTK 天线 (例如 left_gnss_link)")]
    public Transform rtkAntennaLeft;
    [Tooltip("左侧 RTK 话题")]
    public string rtkLeftTopic = "/rtk_left/fix";

    [Tooltip("右侧 RTK 天线 (例如 right_gnss_link)")]
    public Transform rtkAntennaRight;
    [Tooltip("右侧 RTK 话题")]
    public string rtkRightTopic = "/rtk_right/fix";

    [Header("3. Map Origin (WGS84 Reference)")]
    [Tooltip("地图原点纬度 (Degrees)")]
    public double refLatitudeDeg = 30.0;
    [Tooltip("地图原点经度 (Degrees)")]
    public double refLongitudeDeg = 120.0;
    [Tooltip("地图原点高程 (Meters)")]
    public double refAltitudeM = 0.0;

    [Tooltip("是否发布 odom_origin (用于显示地图原点的 UTM 坐标)")]
    public bool publishOdomOrigin = true;
    public string odomOriginTopic = "/odom_origin";

    [Header("4. Safety Limits (安全阈值)")]
    [Tooltip("如果计算出的速度超过此值(m/s)，视为瞬移，强制归零。防止ROS飞车。")]
    public float maxSpeedThreshold = 5.0f; 
    [Tooltip("如果速度小于此值(m/s)，视为静止。防止物理引擎微颤导致地图漂移。")]
    public float minMoveThreshold = 0.01f;

    [Header("Frequency")]
    public float publishHz = 50f;

    // ================= 【新增：时间策略设置】 =================
    [Tooltip("可选：拖入实现 Now(): TimeMsg 的 Clock 组件（如 ROS Clock）。留空则使用 Time.time。")]
    public MonoBehaviour clockSource;
    // =======================================================

    // --- 内部变量 ---
    private ROSConnection _ros;
    private float _publishInterval;
    private float _timeSinceLastPublish;

    // Odom 相对原点记录
    private Vector3 _odomOriginPos;
    private bool _odomOriginSet = false;

    // UTM 变量 (用于 odom_origin)
    private double _utmOriginX;
    private double _utmOriginY;

    // 用于微分计算的缓存
    private Vector3 _lastFramePos;
    private Quaternion _lastFrameRot;
    private float _lastFrameTime;
    private bool _hasLastFrameData = false;

    // 数学常量
    const double Deg2Rad = Math.PI / 180.0;
    const double Rad2Deg = 180.0 / Math.PI;
    const double EarthRadius = 6378137.0;

    private void Start()
    {
        _ros = ROSConnection.GetOrCreateInstance();
        _publishInterval = publishHz > 0f ? 1.0f / publishHz : 0.02f;

        // // 自动查找组件：如果用户没拖 Rigidbody，尝试在 baseFootprint 上找 AGX 的刚体
        // if (trackRigidbody == null && baseFootprint != null)
        // {
        //     // 【关键修改 3】查找 AGXUnity.RigidBody 组件
        //     trackRigidbody = baseFootprint.GetComponent<AGXUnity.RigidBody>();
        //     if (trackRigidbody == null)
        //     {
        //         // 尝试在子物体里找（有时刚体在子节点）
        //         trackRigidbody = baseFootprint.GetComponentInChildren<AGXUnity.RigidBody>();
        //     }

        //     if (trackRigidbody == null)
        //     {
        //         Debug.LogWarning("[Excavator922FOdomPublisher] 未指定 trackRigidbody，且 baseFootprint 上也没有 AGX RigidBody。速度将无法正确发布！");
        //     }
        // }

        // 注册 ROS 话题
        _ros.RegisterPublisher<OdometryMsg>(odomTopic);
        if (publishOdomOrigin)
        {
            _ros.RegisterPublisher<OdometryMsg>(odomOriginTopic);
        }

        if (rtkAntennaLeft != null)
        {
            _ros.RegisterPublisher<NavSatFixMsg>(rtkLeftTopic);
        }
        
        if (rtkAntennaRight != null)
        {
            _ros.RegisterPublisher<NavSatFixMsg>(rtkRightTopic);
        }        

        // 初始化 UTM 原点 (将参考经纬度转换为 UTM)
        int zone;
        bool isNorthern;
        LatLonToUTM(refLatitudeDeg * Deg2Rad, refLongitudeDeg * Deg2Rad, out _utmOriginX, out _utmOriginY, out zone, out isNorthern);
        Debug.Log($"[Excavator922FOdomPublisher] Map Origin initialized. UTM Zone: {zone}{(isNorthern ? "N" : "S")}");
    }

    private void FixedUpdate()
    {
        _timeSinceLastPublish += Time.fixedDeltaTime;
        if (_timeSinceLastPublish < _publishInterval)
            return;

        _timeSinceLastPublish = 0f;

        // 使用整合了 Time Strategy 的获取时间戳方法
        var stamp = GetSimTimeStamp();

        // 1. 发布底盘里程计
        PublishOdom(stamp);

        // 2. 发布天线定位
        PublishRtkFix(stamp);
    }

    /// <summary>
    /// 核心方法：读取 base_footprint 的状态并转换为 ROS Odom 消息
    /// </summary>
    void PublishOdom(TimeMsg stamp)
    {
        if (baseFootprint == null) 
            return;

        // A. 获取当前 Unity 世界坐标系下的位姿
        Vector3 currentPos = baseFootprint.position;
        Quaternion currentRot = baseFootprint.rotation;
        float currentTime = Time.time;

        // B. 设定初始原点 (第一次运行时的位置作为 (0,0,0))
        if (!_odomOriginSet)
        {
            _odomOriginPos = currentPos;
            _odomOriginSet = true;
        }

        // C. 计算相对位移 (Unity 坐标系)
        Vector3 relPos = currentPos - _odomOriginPos;

        // ================= 【核心修改 1：手动微分计算速度】 =================
        // 目的：强制让速度与位移 100% 同步，解决 "RViz 追不上" 的问题。

        // D. 获取速度 (Twist)
        Vector3 linVel = Vector3.zero;
        Vector3 angVel = Vector3.zero;

        if (_hasLastFrameData)
        {
            float dt = currentTime - _lastFrameTime;

            // [安全保护] 防止除以极小值
            if (dt > 1e-5f)
            {
                // 1. 计算世界坐标系下的原始速度 (位移 / 时间)
                Vector3 worldLinVel = (currentPos - _lastFramePos) / dt;

                // [安全保护] 防瞬移 (Anti-Teleport)
                if (worldLinVel.magnitude > maxSpeedThreshold)
                {
                    worldLinVel = Vector3.zero;
                }

                // [安全保护] 防微颤 (Deadzone)
                if (worldLinVel.magnitude < minMoveThreshold)
                {
                    worldLinVel = Vector3.zero;
                }

                // 2. 转到局部坐标系 (ROS Twist 要求 Child Frame 下的速度)
                linVel = baseFootprint.InverseTransformDirection(worldLinVel);

                // 3. 计算角速度 (四元数微分)
                Quaternion deltaRot = Quaternion.Inverse(_lastFrameRot) * currentRot;
                deltaRot.ToAngleAxis(out float angle, out Vector3 axis);
                if (angle > 180f)
                {
                    angle -= 360f; // 归一化到 -180 ~ 180
                }
                    
                // 角速度死区保护
                if (Mathf.Abs(angle) < 0.01f) 
                {
                    angle = 0;
                }

                angVel = axis * (angle * Mathf.Deg2Rad / dt);               
            }
        }

        // 更新缓存
        _lastFramePos = currentPos;
        _lastFrameRot = currentRot;
        _lastFrameTime = currentTime;
        _hasLastFrameData = true;

        // if (trackRigidbody != null)
        // {
        //     // 【关键修改 4】使用 AGX 的 API 读取速度 (LinearVelocity / AngularVelocity)
        //     Vector3 worldLinVel = trackRigidbody.LinearVelocity;
        //     Vector3 worldAngVel = trackRigidbody.AngularVelocity;

        //     // 转换为 base_footprint 的局部坐标系 (Robot Body Frame)
        //     // 这一点非常重要：ROS 的 nav_msgs/Odometry 中的 twist.twist 通常是 Child Frame (base_footprint) 下的速度
        //     linVel = baseFootprint.InverseTransformDirection(worldLinVel);
        //     angVel = baseFootprint.InverseTransformDirection(worldAngVel);
        // }

        // ================= 【核心修改：针对 Forward = -X 的映射】 =================

        // 【1. 位置转换】
        // 你的现状：Unity 向前移动 = X 轴数值减小 (比如 -1 到 -3)
        // 修正逻辑：ROS X (前) = -Unity X (负负得正)
        //          ROS Y (左) = -Unity Z (右手定则补充)
        //          ROS Z (上) = Unity Y 
        var rosPos = new PointMsg(-relPos.x, -relPos.z, relPos.y);

        // 【2. 旋转转换】
        // 必须与位置映射保持一致：(-x, -z, y, -w)
        Quaternion rawRosRot = new Quaternion(-currentRot.x, -currentRot.z, currentRot.y, -currentRot.w);

        // 静态姿态矫正
        // 建议先设为 identity (无修正)，看看在新的轴向映射下，车头和车身是否已经正常。
        // 如果运行后发现车头对了但"竖起来"了，再尝试改为 Quaternion.Euler(0, 0, 90) 或其他值。
        // Quaternion fixRot = Quaternion.identity;
        Quaternion fixRot = Quaternion.Euler(90, 0, 0);

        Quaternion finalRosRot = rawRosRot * fixRot;

        var rosRotMsg = new QuaternionMsg(finalRosRot.x, finalRosRot.y, finalRosRot.z, finalRosRot.w);

        // 【3. 速度转换】
        // 映射规则同位置：X -> X
        // 因为我们上面手动算出的 linVel 已经是局部坐标系下的前后左右了，
        // 这里只需要处理轴向定义：Unity X轴对应 ROS X轴。
        // 注意：InverseTransformDirection 得到的 linVel.x 对应的是 Unity 自身的 Right，
        // 由于你的模型旋转了 -90度，这里的转换需要特别小心。
        // 但根据 rosPos 的逻辑 (-x, -z, y)，速度通常遵循同样的符号规则。
        var rosLinVel = new Vector3Msg(-linVel.x, -linVel.z, linVel.y);
        var rosAngVel = new Vector3Msg(-angVel.x, -angVel.z, angVel.y);

        // F. 组装消息
        var odomMsg = new OdometryMsg
        {
            header = new HeaderMsg
            {
                stamp = stamp,
                frame_id = odomFrameId
            },
            child_frame_id = baseFrameId,
            pose = new PoseWithCovarianceMsg
            {
                // pose = new PoseMsg(rosPos, rosRot),
                pose = new PoseMsg(rosPos, rosRotMsg), // 使用修正后的旋转
                covariance = new double[36] // 协方差暂空
            },
            twist = new TwistWithCovarianceMsg
            {
                twist = new TwistMsg(rosLinVel, rosAngVel),
                covariance = new double[36]
            }
        };

        _ros.Publish(odomTopic, odomMsg);

        // G. (可选) 发布 Origin
        if (publishOdomOrigin && _odomOriginSet)
        {
            // PublishOdomOrigin(stamp, rosRot);
            PublishOdomOrigin(stamp, rosRotMsg);
        }
    }

    // --- 辅助函数保持不变 ---
    void PublishOdomOrigin(TimeMsg stamp, QuaternionMsg rosRot)
    {
        var msg = new OdometryMsg
        {
            header = new HeaderMsg { stamp = stamp, frame_id = "odom_origin" },
            child_frame_id = baseFrameId,
            pose = new PoseWithCovarianceMsg
            {
                // 使用计算出的绝对 UTM 坐标
                pose = new PoseMsg(new PointMsg(_utmOriginX, _utmOriginY, refAltitudeM), rosRot),
                covariance = new double[36]
            }
        };
        _ros.Publish(odomOriginTopic, msg);
    }

    void PublishRtkFix(TimeMsg stamp)
    {
        // 只要天线 Transform 是 body_link 的子物体，position 就会自动跟随驾驶室旋转
        if (rtkAntennaLeft != null)
        {
            var msg = BuildNavSatFix(stamp, rtkAntennaLeft.position, "rtk_left_link");
            _ros.Publish(rtkLeftTopic, msg);
        }

        if (rtkAntennaRight != null)
        {
            var msg = BuildNavSatFix(stamp, rtkAntennaRight.position, "rtk_right_link");
            _ros.Publish(rtkRightTopic, msg);
        }
    }

    NavSatFixMsg BuildNavSatFix(TimeMsg stamp, Vector3 worldPos, string frameId)
    {
        // 将 Unity 世界坐标转换为 ENU (ROS标准: X=East, Y=North, Z=Up)
        // Unity: Z=North(approx), X=East(approx), Y=Up
        double east = worldPos.x;
        double north = worldPos.z;
        double up = worldPos.y;

        double latDeg, lonDeg, altM;
        LocalEnuToWgs84(east, north, up, out latDeg, out lonDeg, out altM);

        return new NavSatFixMsg
        {
            header = new HeaderMsg
            {
                stamp = stamp,
                frame_id = frameId
            },
            status = new NavSatStatusMsg
            {
                status = NavSatStatusMsg.STATUS_FIX,
                service = NavSatStatusMsg.SERVICE_GPS
            },
            latitude = latDeg,
            longitude = lonDeg,
            altitude = altM,
            position_covariance = new double[9],
            position_covariance_type = NavSatFixMsg.COVARIANCE_TYPE_UNKNOWN
        };
    }

    // --- 数学工具函数 ---

    /// <summary>
    /// 平面近似：局部 ENU 坐标转 WGS84 经纬度
    /// </summary>
    void LocalEnuToWgs84(double east, double north, double up,
                         out double latDeg, out double lonDeg, out double altM)
    {
        double lat0Rad = refLatitudeDeg * Deg2Rad;
        double lon0Rad = refLongitudeDeg * Deg2Rad;

        double dLat = north / EarthRadius;
        double dLon = east / (EarthRadius * Math.Cos(lat0Rad));

        double latRad = lat0Rad + dLat;
        double lonRad = lon0Rad + dLon;

        latDeg = latRad * Rad2Deg;
        lonDeg = lonRad * Rad2Deg;
        altM = refAltitudeM + up;
    }

    /// <summary>
    /// 标准算法：WGS84 经纬度转 UTM 坐标 (完整实现)
    /// </summary>
    void LatLonToUTM(double lat, double lon, out double utmX, out double utmY, out int zone, out bool isNorthern)
    {
        double latDeg = lat * Rad2Deg;
        double lonDeg = lon * Rad2Deg;

        zone = (int)((lonDeg + 180.0) / 6.0) + 1;
        isNorthern = (latDeg >= 0.0);

        const double a = 6378137.0;
        const double f = 1.0 / 298.257223563;
        const double k0 = 0.9996;
        const double e2 = 2 * f - f * f;

        double lon0Deg = (zone - 1) * 6.0 - 180.0 + 3.0;
        double lon0 = lon0Deg * Deg2Rad;

        double N = a / Math.Sqrt(1 - e2 * Math.Sin(lat) * Math.Sin(lat));
        double T = Math.Tan(lat) * Math.Tan(lat);
        double C = e2 * Math.Cos(lat) * Math.Cos(lat) / (1 - e2);
        double A = Math.Cos(lat) * (lon - lon0);
        double M = a * ((1 - e2 / 4 - 3 * e2 * e2 / 64) * lat
                        - (3 * e2 / 8 + 3 * e2 * e2 / 32) * Math.Sin(2 * lat)
                        + (15 * e2 * e2 / 256) * Math.Sin(4 * lat));

        utmX = k0 * N * (A + (1 - T + C) * A * A * A / 6 + (5 - 18 * T + T * T + 72 * C) * A * A * A * A * A / 120) + 500000.0;
        utmY = k0 * (M + N * Math.Tan(lat) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A * A * A * A / 24 + (61 - 58 * T + T * T) * A * A * A * A * A * A / 720));

        if (!isNorthern)
        {
            utmY += 10000000.0;
        }
    }

    // ================= 【新增：混合时间戳获取策略】 =================
    /// <summary>
    /// 获取仿真时间戳：
    /// 1. 优先尝试通过反射调用 clockSource 组件的 Now() 方法。
    /// 2. 如果失败或未设置，则回退到 Unity 的 Time.time。
    /// </summary>
    private TimeMsg GetSimTimeStamp()
    {
        // 策略 1: 外部时钟源 (External Clock Source)
        if (clockSource != null)
        {
            var method = clockSource.GetType().GetMethod(
                "Now",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic
            );
            
            if (method != null && method.GetParameters().Length == 0 && method.ReturnType == typeof(TimeMsg))
            {
                try
                {
                    return (TimeMsg)method.Invoke(clockSource, null);
                }
                catch { /* 调用失败则静默回退 */ }
            }
        }

        // 策略 2: Unity 游戏时间 (Fallback)
        float t = Time.time;
        uint sec = (uint)t;
        uint nsec = (uint)((t - sec) * 1e9f);
        return new TimeMsg((int)sec, nsec);
    }
}