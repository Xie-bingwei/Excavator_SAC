using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AGXUnity;
using UnityEngine.InputSystem;

public class ExcavatorE85SceneController : ScriptComponent
{
  public ExcavatorE85CameraHandler CameraHandler
  {
    get
    {
      if (m_cameraHandler == null)
        m_cameraHandler = GetComponent<ExcavatorE85CameraHandler>();
      return m_cameraHandler;
    }
  }

  [Header("InputAction")]
  [SerializeField]
  private InputActionAsset m_inputAsset = null;

  public InputActionAsset InputAsset
  {
    get { return m_inputAsset; }
    set
    {
      m_inputAsset = value;
      InputMap = m_inputAsset?.FindActionMap("Scene Inputs");
      InputMap.Enable();
    }
  }

  public InputActionMap InputMap = null;

  void Update()
  {
    if (InputMap["Quit"].IsPressed())
      Application.Quit();

    if (InputMap["SwitchCamera"].WasPressedThisFrame())
      CameraHandler.SwitchCamera();

    if (InputMap["FPSCamera"].WasPressedThisFrame())
      CameraHandler.ToggleFPSCamera();

    if (InputMap["Reset"].WasPerformedThisFrame())
      ResetScene();
  }

  public void ResetScene()
  {
    // Reset terrain
    var terrain = GetComponentInChildren<AGXUnity.Model.DeformableTerrain>();
    if (terrain != null)
      terrain.ResetHeights();
  }

  private ExcavatorE85CameraHandler m_cameraHandler = null;
}
