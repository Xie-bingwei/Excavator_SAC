using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AGXUnity;

#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public class ResetTerrain : MonoBehaviour
{

#if ENABLE_INPUT_SYSTEM
  private InputAction ResetAction;
#else
  public KeyCode ResetTerrainKey = KeyCode.R;
#endif

  // Start is called before the first frame update
  void Start()
  {
#if ENABLE_INPUT_SYSTEM
    ResetAction = new InputAction("Reset", binding: "<Keyboard>/r");
    ResetAction.Enable();
#endif

  }

  // Update is called once per frame
  void Update()
  {
#if ENABLE_INPUT_SYSTEM
    if (ResetAction.triggered)
#else
    if (Input.GetKeyDown(ResetTerrainKey))
#endif
    {
      var terrain = GetComponent<AGXUnity.Model.DeformableTerrain>();
      if (terrain != null)
        terrain.ResetHeights();
    }
  }
}

