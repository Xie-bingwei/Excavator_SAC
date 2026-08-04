using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class ExcavatorE85CameraHandler : MonoBehaviour
{
  public List<Camera> cameras;
  public Camera FPSCamera;

  public void SwitchCamera(string name)
  {
    for(int i=0; i<cameras.Count; i++)
    {
      if (cameras[i].name == name)
      {
        SwitchToCamera(cameras[i]);
        m_activeCameraIndex = i;
      }
    }
  }

  public void SwitchCamera()
  {
    if (FPSCamera != null)
      FPSCamera.gameObject.SetActive(false);

    m_activeCameraIndex++;
    if (m_activeCameraIndex >= cameras.Count)
      m_activeCameraIndex = 0;

    SwitchToCamera(cameras[m_activeCameraIndex]);
  }

  public void ToggleFPSCamera()
  {
    if (!FPSCamera.gameObject.activeSelf)
      SwitchToCamera(FPSCamera);
    else
      SwitchToCamera(cameras[m_activeCameraIndex]);
  }

  private void SwitchToCamera(Camera camera)
  {
    m_currentActive.gameObject.SetActive(false);
    m_currentActive = camera;
    camera.gameObject.SetActive(true);
  }

  private void Start()
  {
    m_currentActive = cameras[0];
    for (int i = 1; i < cameras.Count; i++)
      cameras[i].gameObject.SetActive(false);

    m_currentActive.gameObject.SetActive(true);

    if (FPSCamera != null)
      FPSCamera.gameObject.SetActive(false);
  }

  private Camera m_currentActive = null;
  private int m_activeCameraIndex = 0;
}
