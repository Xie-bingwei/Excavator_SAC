using UnityEngine;
using UnityEditor;
using AGXUnity;
using AGXUnity.Collide;

public class SetupPipeCollider
{
    [MenuItem("Tools/Setup Pipe Collider")]
    public static void Setup()
    {
        var pipe = GameObject.Find("underground_pipe");
        if (pipe == null)
        {
            Debug.LogError("underground_pipe not found in scene!");
            return;
        }

        // 移除原生 Unity collider
        var uc = pipe.GetComponent<Collider>();
        if (uc != null) Object.DestroyImmediate(uc);

        // AGX RigidBody (Static)
        var rb = pipe.GetComponent<RigidBody>();
        if (rb == null) rb = Undo.AddComponent<RigidBody>(pipe.gameObject);
        rb.MotionControl = agx.RigidBody.MotionControl.STATIC;

        // AGX Cylinder
        var cyl = pipe.GetComponent<Cylinder>();
        if (cyl == null) cyl = Undo.AddComponent<Cylinder>(pipe.gameObject);
        cyl.Radius = 0.2f;
        cyl.Height = 6.0f;

        // AGX CollisionGroups
        var cg = pipe.GetComponent<CollisionGroups>();
        if (cg == null) cg = Undo.AddComponent<CollisionGroups>(pipe.gameObject);

        Debug.Log("Pipe: AGX RigidBody(STATIC) + Cylinder(r=0.2, h=6.0) + CollisionGroups setup complete");
    }
}
