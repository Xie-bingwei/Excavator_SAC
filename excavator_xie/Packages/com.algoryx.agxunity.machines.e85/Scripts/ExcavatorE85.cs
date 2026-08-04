using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AGXUnity;

public class ExcavatorE85 : ScriptComponent
{
  public enum Location
  {
    Right,
    Left,
  }

  public float Speed
  {
    get
    {
      if (m_underCarriageObserver == null)
        m_underCarriageObserver = transform.Find("UnderCarriageBody/UnderCarriageObserver").GetComponent<ObserverFrame>();
      if (m_underCarriageObserver == null || m_underCarriageObserver.Native == null)
        return 0.0f;

      var v = m_underCarriageObserver.Native.getVelocity();
      v = m_underCarriageObserver.Native.transformVectorToLocal(v);
      return (float)(3.6 * v[1]);
    }
  }

  [AllowRecursiveEditing]
  public Constraint ArmPrismatic
  {
    get
    {
      if (m_armPrismatic == null)
        m_armPrismatic = transform.Find("ArmPrismatic").GetComponent<Constraint>();
      return m_armPrismatic;
    }
  }

  [AllowRecursiveEditing]
  public Constraint SwingPrismatic
  {
    get
    {
      if (m_swingPrismatic == null)
        m_swingPrismatic = transform.Find("ArticulatedArm_Prismatic").GetComponent<Constraint>();
      return m_swingPrismatic;
    }
  }

  [AllowRecursiveEditing]
  public Constraint StickPrismatic
  {
    get
    {
      if (m_stickPrismatic == null)
        m_stickPrismatic = transform.Find("StickPrismatic").GetComponent<Constraint>();
      return m_stickPrismatic;
    }
  }

  [AllowRecursiveEditing]
  public Constraint BucketPrismatic
  {
    get
    {
      if (m_bucketPrismatic == null)
        m_bucketPrismatic = transform.Find("BucketPrismatic").GetComponent<Constraint>();
      return m_bucketPrismatic;
    }
  }

  [AllowRecursiveEditing]
  public Constraint TiltPrismatic
  {
    get
    {
      if (m_tiltPrismatic == null)
        m_tiltPrismatic = transform.Find("TiltPrismatic").GetComponent<Constraint>();
      return m_tiltPrismatic;
    }
  }

  [AllowRecursiveEditing]
  public Constraint CabinHinge
  {
    get
    {
      if (m_cabinHinge == null)
        m_cabinHinge = transform.Find("CabinHinge").GetComponent<Constraint>();
      return m_cabinHinge;
    }
  }

  [AllowRecursiveEditing]
  public Constraint LeftHinge
  {
    get
    {
      if (m_sprocketHinges[(int)Location.Left] == null)
        m_sprocketHinges[(int)Location.Left] = transform.Find("LeftSprocketHinge").GetComponent<Constraint>();
      return m_sprocketHinges[(int)Location.Left];
    }
  }

  [AllowRecursiveEditing]
  public Constraint RightHinge
  {
    get
    {
      if (m_sprocketHinges[(int)Location.Right] == null)
        m_sprocketHinges[(int)Location.Right] = transform.Find("RightSprocketHinge").GetComponent<Constraint>();
      return m_sprocketHinges[(int)Location.Right];
    }
  }

  [HideInInspector]
  public Constraint[] BladePrismatics
  {
    get
    {
      if (m_bladePrismatics[(int)Location.Left] == null)
        m_bladePrismatics[(int)Location.Left] = transform.Find("BladePrismatic2").GetComponent<Constraint>();

      if (m_bladePrismatics[(int)Location.Right] == null)
        m_bladePrismatics[(int)Location.Right] = transform.Find("BladePrismatic1").GetComponent<Constraint>();
      return m_bladePrismatics;
    }
  }

  private ObserverFrame m_underCarriageObserver = null;
  private Constraint[] m_sprocketHinges = new Constraint[] { null, null };

  private Constraint m_swingPrismatic = null;
  private Constraint m_armPrismatic = null;
  private Constraint m_stickPrismatic = null;
  private Constraint m_bucketPrismatic = null;
  private Constraint m_tiltPrismatic = null;
  private Constraint m_cabinHinge = null;
  private Constraint[] m_bladePrismatics = new Constraint[] { null, null };
}
