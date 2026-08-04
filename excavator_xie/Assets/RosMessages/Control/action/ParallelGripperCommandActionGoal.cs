using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Std;
using RosMessageTypes.Actionlib;

namespace RosMessageTypes.Control
{
    public class ParallelGripperCommandActionGoal : ActionGoal<ParallelGripperCommandGoal>
    {
        public const string k_RosMessageName = "control_msgs/ParallelGripperCommandActionGoal";
        public override string RosMessageName => k_RosMessageName;


        public ParallelGripperCommandActionGoal() : base()
        {
            this.goal = new ParallelGripperCommandGoal();
        }

        public ParallelGripperCommandActionGoal(HeaderMsg header, GoalIDMsg goal_id, ParallelGripperCommandGoal goal) : base(header, goal_id)
        {
            this.goal = goal;
        }
        public static ParallelGripperCommandActionGoal Deserialize(MessageDeserializer deserializer) => new ParallelGripperCommandActionGoal(deserializer);

        ParallelGripperCommandActionGoal(MessageDeserializer deserializer) : base(deserializer)
        {
            this.goal = ParallelGripperCommandGoal.Deserialize(deserializer);
        }
        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.header);
            serializer.Write(this.goal_id);
            serializer.Write(this.goal);
        }


#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
#else
        [UnityEngine.RuntimeInitializeOnLoadMethod]
#endif
        public static void Register()
        {
            MessageRegistry.Register(k_RosMessageName, Deserialize);
        }
    }
}
