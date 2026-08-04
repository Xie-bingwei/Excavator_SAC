using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

// 让一个摄像头的镜头绑在挖掘机的某个部件

#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif


[RequireComponent(typeof(Camera))]
public class LinkCamera : MonoBehaviour
{
  [SerializeField]
  public bool Enabled = true;

#if ENABLE_INPUT_SYSTEM
  private InputAction EnableAction;
#else
  [SerializeField]
  public KeyCode toggleEnableKey = KeyCode.F1;
#endif

  public Vector3 Forward = new Vector3(0, 0, 1);
  public float Distance = 0.5f;
  public Vector3 RelativePosition;


  [SerializeField]
  private GameObject m_follow_object = null;

  public GameObject Target
  {
    get { return m_follow_object; }
    set
    {
      m_follow_object = value;
    }
  }

  public LinkCamera()
  {
  }

  // Start is called before the first frame update
  void Start()
  {
#if ENABLE_INPUT_SYSTEM
    EnableAction = new InputAction("Enable", binding: "<Keyboard>/F1");
    EnableAction.Enable();
#endif
  }

  void OnGUI()
  {
    Update();
  }

  // Update is called once per frame
  void Update()
  {
    if (Target == null)
      return;

#if ENABLE_INPUT_SYSTEM
    if (EnableAction.triggered)
#else
    if (Input.GetKeyDown(toggleEnableKey))
#endif
      Enabled = !Enabled;

    if (!Enabled)
      return;


    var forward = Target.transform.TransformVector(Forward);
    forward.Normalize();

    var dir = Target.transform.TransformVector(RelativePosition);

    transform.position = Target.transform.position + dir;

    var lookAt = transform.position + forward * 1.0f;
    transform.LookAt(lookAt);
  }
}
