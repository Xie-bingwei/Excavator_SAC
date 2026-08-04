using System.Collections.Generic;
using UnityEngine;
using AGXUnity;
using System.IO;

/// <summary>
/// 齿尖轨迹录制器 — 示教模式
/// 进 Play 模式后按 1 开始录制，按 2 停止并导出 JSON。
/// 录制挂在 AGX 步进回调上，确保每一步物理后都记录数据。
/// </summary>
public class TipTrajectoryRecorder : ScriptComponent
{
    [Header("Output")]
    [Tooltip("导出文件路径 (相对于项目根目录)")]
    public string outputPath = "recorded_trajectory.json";

    [Header("Status")]
    [SerializeField]
    private bool _isRecording = false;
    [SerializeField]
    private int _frameCount = 0;

    private readonly List<RecordedFrame> _frames = new List<RecordedFrame>();
    private Excavator922FJointStatePublisher _publisher;

    [System.Serializable]
    public struct RecordedFrame
    {
        public float time;
        public double[] q_ros;
        public float[] p_tip;
        public float[] p_tip_bf;
    }

    protected override void OnEnable()
    {
        _publisher = GetComponent<Excavator922FJointStatePublisher>();
        if (_publisher == null)
            _publisher = GetComponentInChildren<Excavator922FJointStatePublisher>();

        Simulation.Instance.StepCallbacks.PostStepForward += OnPostStep;
    }

    protected override void OnDisable()
    {
        Simulation.Instance.StepCallbacks.PostStepForward -= OnPostStep;
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1) && !_isRecording)
        {
            _frames.Clear();
            _frameCount = 0;
            _isRecording = true;
            Debug.Log("[TipRecorder] 开始录制齿尖轨迹... 按 2 停止");
        }

        if (Input.GetKeyDown(KeyCode.Alpha2) && _isRecording)
        {
            _isRecording = false;
            Debug.Log($"[TipRecorder] 录制停止，共 {_frameCount} 帧");
            ExportToJson();
        }
    }

    private void OnPostStep()
    {
        if (!_isRecording) return;
        RecordFrame();
    }

    private void RecordFrame()
    {
        var tip = GameObject.Find("bucket_tip_link");
        var bf  = GameObject.Find("base_footprint");
        if (tip == null || bf == null) return;

        Vector3 tipWorld = tip.transform.position;
        Vector3 tipBF    = bf.transform.InverseTransformPoint(tipWorld);

        var qRos = new double[4];
        if (_publisher != null && _publisher.robotJoints != null)
        {
            for (int i = 0; i < _publisher.robotJoints.Count && i < 4; i++)
            {
                var cfg = _publisher.robotJoints[i];
                qRos[i] = cfg.agxConstraint.GetCurrentAngle() * cfg.multiplier + cfg.offset;
            }
        }

        _frames.Add(new RecordedFrame
        {
            time  = Time.time,
            q_ros = qRos,
            p_tip = new float[] { tipWorld.x, tipWorld.y, tipWorld.z },
            p_tip_bf = new float[] { tipBF.x, tipBF.y, tipBF.z },
        });

        _frameCount++;
    }

    private void ExportToJson()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("[");
        for (int i = 0; i < _frames.Count; i++)
        {
            var f = _frames[i];
            sb.Append("  {");
            sb.Append($"\"t\":{f.time:F3}, ");
            sb.Append($"\"q\":[{f.q_ros[0]:F6},{f.q_ros[1]:F6},{f.q_ros[2]:F6},{f.q_ros[3]:F6}], ");
            sb.Append($"\"p_world\":[{f.p_tip[0]:F4},{f.p_tip[1]:F4},{f.p_tip[2]:F4}], ");
            sb.Append($"\"p_bf\":[{f.p_tip_bf[0]:F4},{f.p_tip_bf[1]:F4},{f.p_tip_bf[2]:F4}]");
            sb.Append("}");
            if (i < _frames.Count - 1) sb.Append(",");
            sb.AppendLine();
        }
        sb.AppendLine("]");

        string fullPath = Path.Combine(Application.dataPath, outputPath);
        fullPath = Path.GetFullPath(fullPath);
        try
        {
            string dir = Path.GetDirectoryName(fullPath);
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(fullPath, sb.ToString());
            Debug.Log($"[TipRecorder] 已导出 {_frames.Count} 帧 → {fullPath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[TipRecorder] 导出失败: {e.Message}");
        }
    }
}
