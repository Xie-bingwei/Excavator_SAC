using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AGXUnity;
using UnityEngine.InputSystem;

/// <summary>
/// 922F 控制脚本。
/// - 底盘：左右履带（left_track / right_track）
/// - 上车体：回转、大臂、斗杆、铲斗（通过约束关节控制）
/// 输入映射沿用 E85 的思路：Drive / Steer / SteerLeft / SteerRight / Swing / Arm / Stick / Bucket ...
/// </summary>
public class Excavator922FTrackController : ScriptComponent
{
  [Header("Track hinge constraints")]
  [Tooltip("左侧履带驱动关节（例如 left_track 上的 Hinge/Constraint）。")]
  public Constraint LeftTrackHinge;

  [Tooltip("右侧履带驱动关节（例如 right_track 上的 Hinge/Constraint）。")]
  public Constraint RightTrackHinge;

  [Header("Upper structure constraints")]
  [Tooltip("上车体回转关节（base_to_body_joint 对应的 Constraint）。")]
  public Constraint SwingConstraint;

  [Tooltip("大臂关节（body_to_boom_joint 对应的 Constraint）。")]
  public Constraint BoomConstraint;

  [Tooltip("斗杆关节（boom_to_arm_joint 对应的 Constraint）。")]
  public Constraint ArmConstraint;

  [Tooltip("铲斗关节（arm_to_bucket_joint 对应的 Constraint）。")]
  public Constraint BucketConstraint;

  [Header("Max speeds (rad/s)")]
  [SerializeField]
  public float MaxTrackSprocketSpeed = 2.27f;

  [SerializeField]
  public float MaxSwingSpeed = 0.15f;

  [SerializeField]
  public float MaxBoomSpeed = 0.1f;

  [SerializeField]
  public float MaxArmSpeed = 0.2f;

  [SerializeField]
  public float MaxBucketSpeed = 0.2f;

  [Header("Debug")]
  [Tooltip("启用后，在控制台输出输入值和约束状态，便于排查问题。")]
  public bool EnableDebugLog = false;

  [Header("Motor forces")]
  [Tooltip("履带驱动电机的力矩范围。")]
  public float TrackMotorForce = 100000f;

  [Tooltip("上车体回转电机的力矩范围。")]
  public float SwingMotorForce = 80000f;

  [Tooltip("大臂电机的力矩范围。")]
  public float BoomMotorForce = 120000f;

  [Tooltip("斗杆电机的力矩范围。")]
  public float ArmMotorForce = 80000f;

  [Tooltip("铲斗电机的力矩范围。")]
  public float BucketMotorForce = 60000f;

  // --- 【新增】ROS 控制信号缓存 ---
  private bool _useRosTrack = false;
  private float _rosLeftSpeed = 0f;
  private float _rosRightSpeed = 0f;

  // 手臂部分变量
  private bool _useRosArm = false;
  private float _rosSwingPos = 0f;
  private float _rosBoomPos = 0f;
  private float _rosArmPos = 0f;
  private float _rosBucketPos = 0f;
  // ------------------------------------

  public enum ActionType
  {
    SteerRight,
    SteerLeft,
    Drive,
    Steer,
    Bucket,
    Arm,
    Boom,
    Tilt,
    Cabin,
    Blade,
    Swing,
    Reset
  }

  [Header("InputAction")]
  [SerializeField]
  private InputActionAsset m_inputAsset = null;

  [Tooltip("输入 ActionMap 名称，建议新建为 \"Excavator922F\"，也可以直接复用现有的 \"ExcavatorE85\"。")]
  [SerializeField]
  private string m_actionMapName = "Excavator922F";

  /// <summary>
  /// 当前使用的 ActionMap。
  /// </summary>
  [HideInInspector]
  public InputActionMap InputMap = null;

  /// <summary>
  /// 暴露给 Inspector 的 InputAsset 属性，设置时会自动查找并启用 ActionMap。
  /// </summary>
  public InputActionAsset InputAsset
  {
    get { return m_inputAsset; }
    set
    {
      m_inputAsset = value;
      SetupInputActionMap();
    }
  }

  [HideInInspector]
  public float SteerLeft { get { return GetValue(ActionType.SteerLeft); } }

  [HideInInspector]
  public float SteerRight { get { return GetValue(ActionType.SteerRight); } }

  [HideInInspector]
  public float Drive { get { return GetValue(ActionType.Drive); } }

  [HideInInspector]
  public float Steer { get { return GetValue(ActionType.Steer); } }

  [HideInInspector]
  public float Bucket { get { return GetValue(ActionType.Bucket); } }

  [HideInInspector]
  public float Arm { get { return GetValue(ActionType.Arm); } }

  [HideInInspector]
  public float Boom { get { return GetValue(ActionType.Boom); } }

  [HideInInspector]
  public float Swing { get { return GetValue(ActionType.Swing); } }

  private bool m_hasValidInputActionMap = false;

  protected override bool Initialize()
  {
    // 初始化时尝试根据 Inspector 中设置的 InputAsset / ActionMapName 建立映射。
    if (m_inputAsset != null)
      SetupInputActionMap();

    return true;
  }

  protected override void OnEnable()
  {
    if (InputMap != null && m_hasValidInputActionMap)
      InputMap.Enable();
  }

  protected override void OnDisable()
  {
    if (InputMap != null)
      InputMap.Disable();
  }

  // --- 【新增】供 RosCmdVelSubscriber 调用的接口 ---
  public void SetRosTrackVelocity(float leftRadS, float rightRadS)
  {
      _useRosTrack = true;
      _rosLeftSpeed = leftRadS;
      _rosRightSpeed = rightRadS;
  }

  // --- 【新增】供 RosJointCmdSubscriber 调用的接口 ---
  public void SetRosSwing(float pos)
  {
      _useRosArm = true;
      _rosSwingPos = pos;
  }
  public void SetRosBoom(float pos)
  {
    _useRosArm = true;
    _rosBoomPos = pos;
  }
  public void SetRosArm(float pos)
  {
    _useRosArm = true;
    _rosArmPos = pos;
  }
  public void SetRosBucket(float pos)
  {
    _useRosArm = true;
    _rosBucketPos = pos;
  }
  // ------------------------------------------

  private void Update()
  {
    if (!m_hasValidInputActionMap)
      return;

    
    float leftSpeed = 0.0f;
    float rightSpeed = 0.0f;

    // 一 检查是否有手动输入 (键盘/手柄)
    // 和 E85 相同的差速逻辑：
    // 1. 优先使用 Steer（游戏手柄左摇杆 X 等）进行原地转向。
    // 2. 否则使用 Drive（W/S 或摇杆 Y）前进/后退。
    // 3. 否则使用 SteerLeft/SteerRight 分别控制两条履带（键盘单独控制）。
    var steer = Steer;
    var drive = Drive;
    bool hasManualInput = !Mathf.Approximately(steer, 0) || !Mathf.Approximately(drive, 0) || 
                          !Mathf.Approximately(SteerLeft, 0) || !Mathf.Approximately(SteerRight, 0);


    // 2. 优先级逻辑：手动 > ROS
    if (hasManualInput)
    {
      _useRosTrack = false; // 一旦手动介入，暂时取消 ROS 托管
      if (!Mathf.Approximately(steer, 0.0f))
      {
      rightSpeed = -steer * MaxTrackSprocketSpeed;
      leftSpeed = steer * MaxTrackSprocketSpeed;
      }
      else if (!Mathf.Approximately(drive, 0.0f))
      {
        rightSpeed = drive * MaxTrackSprocketSpeed;
        leftSpeed = drive * MaxTrackSprocketSpeed;
      }
      else
      {
        rightSpeed = SteerRight * MaxTrackSprocketSpeed;
        leftSpeed = SteerLeft * MaxTrackSprocketSpeed;
      }
    }
    else if (_useRosTrack) // 如果没动键盘，且 ROS 有指令，就用 ROS 的
    {
      rightSpeed = _rosRightSpeed;
      leftSpeed = _rosLeftSpeed;
    }

    // 3. 应用到底盘 AGX 关节
    // 注意：这里调用的是 SetSpeed，里面操作的是 AGX 的 TargetSpeedController
    if (LeftTrackHinge != null)
      SetSpeed(LeftTrackHinge, leftSpeed, TrackMotorForce);

    if (RightTrackHinge != null)
      SetSpeed(RightTrackHinge, rightSpeed, TrackMotorForce);

    // 4. 应用到手臂 AGX 关节

    // --- 【优化建议】检测手臂的手动输入，如果有操作，则打断 ROS 控制 ---
    bool hasArmManualInput = !Mathf.Approximately(Swing, 0) || 
                             !Mathf.Approximately(Boom, 0) || 
                             !Mathf.Approximately(Arm, 0) || 
                             !Mathf.Approximately(Bucket, 0);
    if (hasArmManualInput)
    {
        _useRosArm = false; // 手动介入，取消 ROS 托管
    }

    // ================================= 【修改开始】 =================================
    // 原来的代码没有区分手动和ROS，这里我们加上判断逻辑
    // 如果 _useRosArm 为 true，使用 SetPosition (位置控制)
    // 否则使用 SetSpeed (速度控制)
    if (_useRosArm)
    {
        // MoveIt 自动模式：目标是位置 (弧度)
        if (SwingConstraint != null)
          SetPosition(SwingConstraint, _rosSwingPos, SwingMotorForce);

        if (BoomConstraint != null)
          SetPosition(BoomConstraint, _rosBoomPos, BoomMotorForce);

        if (ArmConstraint != null) 
          SetPosition(ArmConstraint, _rosArmPos, ArmMotorForce);

        if (BucketConstraint != null) 
          SetPosition(BucketConstraint, _rosBucketPos, BucketMotorForce);
    }
    else
    {
      // 键盘/手柄手动模式：速度控制
      if (SwingConstraint != null)
        SetSpeed(SwingConstraint, Swing * MaxSwingSpeed, SwingMotorForce);

      if (BoomConstraint != null)
        SetSpeed(BoomConstraint, Boom * MaxBoomSpeed, BoomMotorForce);

      if (ArmConstraint != null)
        SetSpeed(ArmConstraint, Arm * MaxArmSpeed, ArmMotorForce);

      if (BucketConstraint != null)
        SetSpeed(BucketConstraint, Bucket * MaxBucketSpeed, BucketMotorForce);
    }
    // ================================= 【修改结束】 =================================

    if (EnableDebugLog)
    {
      Debug.Log(
        $"[922F Input] Drive={drive:F2}, Steer={steer:F2}, " +
        $"Boom={Boom:F2}, Arm={Arm:F2}, Bucket={Bucket:F2}, Swing={Swing:F2}");
    }    
  }

  /// <summary>
  /// 根据 ActionType 读取输入值（假定为 [-1, 1] 浮点）。
  /// </summary>
  private float GetValue(ActionType action)
  {
    return m_hasValidInputActionMap && InputMap != null
      ? InputMap[action.ToString()].ReadValue<float>()
      : 0.0f;
  }

  /// <summary>
  /// 给约束设置目标转速，逻辑与 E85InputController 中的 SetSpeed 基本一致。
  /// </summary>
  /// // --- 【核心】AGX 电机控制逻辑 (原版保留) ---
  private void SetSpeed(Constraint constraint, float speed, float force)
  {
    if (constraint == null)
    {
      if (EnableDebugLog)
        Debug.LogWarning("[922F] 尝试设置关节速度，但 Constraint 引用为空。");
      return;
    }

    // 判断是否有输入
    var motorEnable = !AGXUnity.Utils.Math.EqualsZero(speed);
    // 获取 AGX 的控制器
    var mc = constraint.GetController<TargetSpeedController>();
    var lc = constraint.GetController<LockController>();

    if (mc == null || lc == null)
    {
      if (EnableDebugLog)
        Debug.LogWarning($"[922F] Constraint \"{constraint.name}\" 上缺少 TargetSpeedController 或 LockController。");
      return;
    }

    mc.Enable = true;
    lc.Enable = false;

    mc.Speed = speed;
    // 给电机足够的力矩，并稍微加一点柔性 & 阻尼
    mc.ForceRange = new RangeReal(force);
    mc.Compliance = 1e-8f; // AGX 参数：柔性
    mc.Damping = 0.1f;     // AGX 参数：阻尼

    // 如果速度为0，启用位置锁定 (LockController)，模拟液压锁定或刹车
    if (!motorEnable)
    {
      lc.Enable = true;
      lc.Position = constraint.GetCurrentAngle(); // 锁定在当前角度
    }
  }

  // ================================= 【新增函数】 =================================
  /// <summary>
  /// 位置控制模式 (MoveIt/ROS Planning)
  /// </summary>
  private void SetPosition(Constraint constraint, float targetPos, float force)
  {
      if (constraint == null) 
        return;
      var mc = constraint.GetController<TargetSpeedController>();
      var lc = constraint.GetController<LockController>();

      if (mc == null || lc == null) 
        return;

      // 切换到位置控制：禁用速度马达，启用锁定控制器并设定目标位置
      mc.Enable = false;

      lc.Enable = true;
      lc.Position = targetPos; // 设定 MoveIt 发来的目标角度
      lc.ForceRange = new RangeReal(force); // 给予足够的力矩

      // 这里的 Compliance (柔性) 不能太大，否则手臂会软绵绵的
      // 但也不能太小(0)，否则物理引擎会因为“绝对刚性”而抖动
      lc.Compliance = 1e-9f; 
      lc.Damping = 0.1f;
  }
  // ================================= 【新增结束】 =================================


  /// <summary>
  /// 查找并校验 ActionMap，以及所需的 Action 是否存在。
  /// </summary>
  private void SetupInputActionMap()
  {
    m_hasValidInputActionMap = false;
    InputMap = null;

    if (m_inputAsset == null || string.IsNullOrEmpty(m_actionMapName))
      return;

    InputMap = m_inputAsset.FindActionMap(m_actionMapName);
    if (InputMap == null)
    {
      Debug.LogWarning($"InputActionAsset doesn't contain an ActionMap named \"{m_actionMapName}\".");
      return;
    }

    // 检查所有 ActionType 对应的 Action 是否存在。
    foreach (var actionName in System.Enum.GetNames(typeof(ActionType)))
    {
      if (InputMap.FindAction(actionName) == null)
      {
        Debug.LogWarning($"Unable to find Input Action: {m_actionMapName}.{actionName}");
        m_hasValidInputActionMap = false;
        return;
      }
    }

    m_hasValidInputActionMap = true;
    InputMap.Enable();
  }
}


