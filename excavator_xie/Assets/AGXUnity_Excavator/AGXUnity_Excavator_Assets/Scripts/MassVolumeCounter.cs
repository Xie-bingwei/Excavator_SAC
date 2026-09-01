using UnityEngine;
using AGXUnity;
using UnityEngine.UI;
using System.Linq;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

/* 土壤与挖掘过程的物理模拟
 * 1.跟踪铲斗内动态质量
 * 2.检测"传感器几何体"接触到的土壤粒子，将其销毁并累加挖掘质量与体积
 * 3.重置地形(按R键), 恢复到预设形状, 重新开始挖掘
*/

// 能获取的值：m_excavatedVolume，m_excavatedMass，m_massInBucket(用于算法)


#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif


public class MassVolumeCounter : ScriptComponent
{

#if ENABLE_INPUT_SYSTEM
  private InputAction ResetAction;
#else
  public KeyCode ResetTerrainKey = KeyCode.R;
#endif


  agxCollide.Geometry m_geometry;
  public AGXUnity.Model.DeformableTerrainShovel shovel; // 铲斗的铲组件
  public AGXUnity.Model.DeformableTerrain m_terrain;    // 可变地形组件
  Terrain m_unityTerrain;  // Unity内置地形，用于同步高度图

  float m_excavatedVolume = 0; // 累计挖掘体积
  float m_excavatedMass = 0;  // 累计挖掘质量
  float m_massInBucket = 0;  // 当前铲斗内的动态质量

 // ── ROS: 在线 RL 奖励信号 ──
  private ROSConnection m_ros;
  private Float64Msg m_volumeMsg     = new Float64Msg();
  private Float64Msg m_bucketMassMsg = new Float64Msg();
  Text m_infoText;

  private agxControl.EventSensor sensor;


  protected override bool Initialize()
  {
    Debug.Assert( m_terrain );
    m_unityTerrain = m_terrain.GetComponent<Terrain>();

#if ENABLE_INPUT_SYSTEM
    ResetAction = new InputAction( "Reset", binding: "<Keyboard>/r" );
    ResetAction.Enable();
#endif

    var texts = GetComponentsInChildren<Text>();
    m_infoText = texts.First( t => t.name == "Information" );

    Debug.Assert( m_infoText );

    // ── ROS: 发布土量 + 订阅地形重置 ──
    m_ros = ROSConnection.GetOrCreateInstance();
    m_ros.RegisterPublisher<Float64Msg>("/unity/soil_volume");
    m_ros.RegisterPublisher<Float64Msg>("/unity/bucket_mass");
    m_ros.Subscribe<BoolMsg>("/unity/reset_terrain", OnResetTerrain);

    return base.Initialize();

  }
  public TerrainData TerrainData { get { return m_unityTerrain?.terrainData; } }


  /// <summary>
  /// Reset the terrain.
  /// Compute a new height for the terrain given some function
  /// </summary>
  void ComputeTerrainHeights()
  {
    m_terrain.ResetHeights();

    var terrain = m_terrain.Native;
    int resX = (int)m_terrain.Native.getResolutionX();
    int resY = (int)m_terrain.Native.getResolutionY();
    double[] height_data = new double[resX * resY];

    // compute new heights for the terrain data
    Vector2 center = new Vector2(resX/2, resY/2);
    for ( var x = 0; x < resX; x++ )
      for ( var y = 0; y < resY; y++ ) {
        var distance = (center - new Vector2(x, y)).magnitude*0.1f;
        var z = (1 / Mathf.Sqrt(2 * Mathf.PI)) * Mathf.Exp(-.5f * distance*distance);
        height_data[ resX * x + y ] = 1 + z * 5;
      }

    // Create a vector we will use to update the terrain heights
    var heights = new agx.RealVector(height_data);

    // update the deformable terrain
    terrain.setHeights( heights );

    // now update the unity terrain
    var scale = TerrainData.heightmapScale.y;
    var result = new float[,] { { 0.0f } };
    for ( var x = 0; x < resX; x++ )
      for ( var y = 0; y < resY; y++ ) {
        var i = (int)x;
        var j = (int)y;
        var h = (float)height_data[resX * x + y];

        result[ 0, 0 ] = h / scale;

        TerrainData.SetHeightsDelayLOD( resX - i - 1, resY - j - 1, result );
      }

#if UNITY_2019_1_OR_NEWER
    TerrainData.SyncHeightmap();
#else
      Terrain.ApplyDelayedHeightmapModification();
#endif



#if UNITY_EDITOR
    // If the editor is closed during play the modified height
    // data isn't saved, this resolves corrupt heights in such case.
    UnityEditor.EditorUtility.SetDirty( TerrainData );
    UnityEditor.AssetDatabase.SaveAssets();
#endif

    // 调整土壤粒子大小缩放因子，影响挖掘时粒子生成的大小
    m_terrain.Native.getProperties().setSoilParticleSizeScaling( 1.5f );


  }

 // ── ROS 重置: 收到 /unity/reset_terrain 时清零并恢复地形 ──
  void OnResetTerrain(BoolMsg msg)
  {
    if (msg.data)
      ResetEpisode();
  }

  void ResetEpisode()
  {
    m_excavatedVolume = 0;
    m_excavatedMass = 0;
    m_terrain.ResetHeights();   // 重置地形到初始形状
  }

  // Update is called once per frame
  void Update()
  {
#if ENABLE_INPUT_SYSTEM

    // If the reset key is pressed.
    if ( ResetAction.triggered )
#else
    if (Input.GetKeyDown(ResetTerrainKey))
#endif
    {
      ResetEpisode();
    }

    Debug.Assert( m_terrain != null );

    // 每帧更新，核心逻辑
    m_massInBucket = (float)shovel.Native.getInnerSoilMass(); // 铲斗内土壤质量(kg)
    string info = string.Format( "Mass in bucket: \t\t{0:f} kg\n", m_massInBucket );

    // 用 AGX 原生 API 直接取土壤量, 不再依赖传感器几何体
    m_excavatedVolume = (float)shovel.Native.getInnerSoilBulkVolume(); // 铲斗内土壤体积 m³
    m_excavatedMass   = (float)shovel.Native.getInnerSoilMass();      // 铲斗内土壤质量 kg

    // Update text
    info += string.Format( "Excavated mass: \t{0:f} kg\n", m_excavatedMass );
    info += string.Format( "Excavated volume: \t{0:f} m^3", m_excavatedVolume );
    m_infoText.text = info;

    // ── ROS: 发布土量 (在线 RL 奖励信号) ──
    m_volumeMsg.data = m_excavatedVolume;
    m_bucketMassMsg.data = m_massInBucket;
    m_ros.Publish("/unity/soil_volume", m_volumeMsg);
    m_ros.Publish("/unity/bucket_mass", m_bucketMassMsg);
  }
}
