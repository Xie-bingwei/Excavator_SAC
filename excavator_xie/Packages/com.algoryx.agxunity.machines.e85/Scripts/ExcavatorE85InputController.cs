using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AGXUnity;
using UnityEngine.InputSystem;

public class ExcavatorE85InputController : ScriptComponent
{
  [Header("Max speeds (m/s or rad/s)")]
  [SerializeField]
  public float MaxTiltSpeed = 0.15f;

  [SerializeField]
  public float MaxBucketSpeed = 0.2f;

  [SerializeField]
  public float MaxStickSpeed = 0.2f;

  [SerializeField]
  public float MaxArmSpeed = 0.1f;

  [SerializeField]
  public float MaxSwingSpeed = 0.1f;

  [SerializeField]
  public float MaxTrackSprocketSpeed = 2.27f;

  [SerializeField]
  public float MaxCabinSpeed = 0.35f;

  [SerializeField]
  public float MaxBladeSpeed = 0.04f;

  public enum ActionType
  {
    Tilt,
    Bucket,
    Stick,
    Arm,
    Swing,
    SteerRight,
    SteerLeft,
    Cabin,
    Blade,
    Drive,
    Steer
  }

  [HideInInspector]
  public ExcavatorE85 Excavator
  {
    get
    {
      if (m_excavator == null)
        m_excavator = GetComponent<ExcavatorE85>();
      return m_excavator;
    }
  }

  [Header("InputAction")]
  [SerializeField]
  private InputActionAsset m_inputAsset = null;

  // Set up input actions
  public InputActionAsset InputAsset
  {
    get
    {
      return m_inputAsset;
    }
    set
    {
      m_inputAsset = value;
      InputMap = m_inputAsset?.FindActionMap( "ExcavatorE85" );

      if (InputMap != null && IsSynchronizingProperties)
      {
        m_hasValidInputActionMap = true;
        foreach (var actionName in System.Enum.GetNames(typeof(ActionType)))
        {
          if (InputMap.FindAction(actionName) == null)
          {
            Debug.LogWarning($"Unable to find Input Action: ExcavatorE85.{actionName}");
            m_hasValidInputActionMap = false;
          }
        }

        if (m_hasValidInputActionMap)
          InputMap.Enable();
        else
          Debug.LogWarning( "ExcavatorE85 input disabled due to missing action(s) in the action map." );
      }

      if (m_inputAsset != null && InputMap == null)
        Debug.LogWarning( "InputActionAsset doesn't contain an ActionMap named \"ExcavatorE85\"." );
    }
  }

  public InputActionMap InputMap = null;

  [HideInInspector]
  public float SteerLeft { get { return GetValue(ActionType.SteerLeft); } }

  [HideInInspector]
  public float SteerRight { get { return GetValue(ActionType.SteerRight); } }

  [HideInInspector]
  public float Cabin { get { return GetValue(ActionType.Cabin); } }

  [HideInInspector]
  public float Tilt { get { return GetValue(ActionType.Tilt); } }

  [HideInInspector]
  public float Bucket { get { return GetValue(ActionType.Bucket); } }

  [HideInInspector]
  public float Stick { get { return GetValue(ActionType.Stick); } }

  [HideInInspector]
  public float Arm { get { return GetValue(ActionType.Arm); } }

  [HideInInspector]
  public float Swing { get { return GetValue(ActionType.Swing); } }

  [HideInInspector]
  public float Blade { get { return GetValue(ActionType.Blade); } }

  [HideInInspector]
  public float Drive { get { return GetValue(ActionType.Drive); } }

  [HideInInspector]
  public float Steer { get { return GetValue(ActionType.Steer); } }


  public float GetValue(ActionType action)
  {
    return m_hasValidInputActionMap ? InputMap[action.ToString()].ReadValue<float>() : 0.0f;
  }

  public void SetSwing(float value)
  {
    SetSpeed(Excavator.SwingPrismatic, value);
  }

  public void SetArm(float value)
  {
    SetSpeed(Excavator.ArmPrismatic, value);
  }

  public void SetStick(float value)
  {
    SetSpeed(Excavator.StickPrismatic, value);
  }

  public void SetBucket(float value)
  {
    SetSpeed(Excavator.BucketPrismatic, value);
  }

  public void SetTilt(float value)
  {
    SetSpeed(Excavator.TiltPrismatic, value);
  }

  public void SetCabin(float value)
  {
    SetSpeed(Excavator.CabinHinge, value);
  }

  public void SetBlade(float value)
  {
    foreach (var prismatic in Excavator.BladePrismatics)
      SetSpeed(prismatic, value);
  }

  public void SetLeftSprocket(float value)
  {
    SetSpeed(Excavator.LeftHinge, value);
  }

  public void SetRightSprocket(float value)
  {
    SetSpeed(Excavator.RightHinge, value);
  }

  public void ResetExcavator()
  {
    // Reset bodies
    for (int i = 0; i < m_bodies.Count; i++)
    {
      m_bodies[i].setTransform(m_transforms[i]);
      m_bodies[i].setVelocity(0.0, 0.0, 0.0);
      m_bodies[i].setAngularVelocity(0.0, 0.0, 0.0);
    }

    //Reset constraints
    foreach (var constraint in m_constraints)
    {
      if (constraint.Type == ConstraintType.Prismatic)
      {
        var prismatic = constraint.Native.asPrismatic();
        bool lockedAtZero = prismatic.getMotor1D().getLockedAtZeroSpeed();
        if (lockedAtZero)
          prismatic.getMotor1D().setLockedAtZeroSpeed(false);
        prismatic.rebind();
        prismatic.getLock1D().setPosition(prismatic.getAngle());
        if (lockedAtZero)
          prismatic.getMotor1D().setLockedAtZeroSpeed(true);
      }
      else if (constraint.Type == ConstraintType.Hinge)
      {
        var hinge = constraint.Native.asHinge();
        bool lockedAtZero = hinge.getMotor1D().getLockedAtZeroSpeed();
        if (lockedAtZero)
          hinge.getMotor1D().setLockedAtZeroSpeed(false);
        hinge.rebind();
        agx.RotationalAngle hingeAngle = agx.RotationalAngle.safeCast(hinge.getAttachmentPair().getAngle(0));
        hingeAngle.setWindingNumber(0);
        hinge.getLock1D().setPosition(hinge.getAngle());
        if (lockedAtZero)
          hinge.getMotor1D().setLockedAtZeroSpeed(true);
      }
    }

    // For the tracks to keep up with the moved bodies, they need to be reinitialzed
    var tracks = Excavator.GetComponentsInChildren<AGXUnity.Model.Track>();
    foreach (var track in tracks)
      track.Native.reinitialize((ulong)track.NumberOfNodes, track.Width, track.Thickness, track.InitialTensionDistance);
  }

  protected override bool Initialize()
  {
    m_constraints = GetComponentsInChildren<Constraint>();
    // Collect all body transforms from start position, to be able to use it for resetting the excavator.
    GetBodiesAndTransforms();
    return true;
  }

  private void Update()
  {
    // Keyboard and gamepad controls the tracks in different ways.
    // Keyboard controls sprockets individually, while the gamepad does controls both at the same time.
    var steer = Steer;
    var drive = Drive;
    if (steer != 0)
    {
      SetRightSprocket(-steer * MaxTrackSprocketSpeed);
      SetLeftSprocket(steer * MaxTrackSprocketSpeed);
    }
    else if (drive != 0)
    {
      SetRightSprocket(drive * MaxTrackSprocketSpeed);
      SetLeftSprocket(drive * MaxTrackSprocketSpeed);
    }
    else
    {
      SetRightSprocket(SteerRight * MaxTrackSprocketSpeed);
      SetLeftSprocket(SteerLeft * MaxTrackSprocketSpeed);
    }

    SetCabin(Cabin * MaxCabinSpeed);
    SetBlade(Blade * MaxBladeSpeed);

    SetBucket(Bucket * MaxBucketSpeed);
    SetTilt(Tilt * MaxTiltSpeed);
    SetStick(Stick * MaxStickSpeed);
    SetArm(Arm * MaxArmSpeed);
    SetSwing(Swing * MaxSwingSpeed);

    if (InputMap["Reset"].WasPerformedThisFrame())
      ResetExcavator();
  }

  private void SetSpeed(Constraint constraint, float speed)
  {
    var motorEnable = !AGXUnity.Utils.Math.EqualsZero(speed);
    var mc = constraint.GetController<TargetSpeedController>();
    var lc = constraint.GetController<LockController>();
    mc.Enable = true;
    lc.Enable = false;

    mc.Speed = speed;

    if (!motorEnable)
    {
      lc.Enable = true;
      lc.Position = constraint.GetCurrentAngle();
    }
  }

  private void GetBodiesAndTransforms()
  {
    var bodies = GetComponentsInChildren<RigidBody>();
    foreach (var body in bodies)
      m_bodies.Add(new agx.RigidBodyRef(body.GetInitialized<RigidBody>().Native));

    m_transforms = new agx.AffineMatrix4x4[m_bodies.Count];
    for (int i = 0; i < m_bodies.Count; i++)
      m_transforms[i] = m_bodies[i].getTransform();
  }

  agx.RigidBodyRefVector m_bodies = new agx.RigidBodyRefVector();
  private agx.AffineMatrix4x4[] m_transforms;
  private Constraint[] m_constraints;

  private ExcavatorE85 m_excavator = null;
  private bool m_hasValidInputActionMap = false;
}
