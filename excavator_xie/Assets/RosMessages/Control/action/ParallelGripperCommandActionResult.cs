using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Std;
using RosMessageTypes.Actionlib;

namespace RosMessageTypes.Control
{
    public class ParallelGripperCommandActionResult : ActionResult<ParallelGripperCommandResult>
    {
        public const string k_RosMessageName = "control_msgs/ParallelGripperCommandActionResult";
        public override string RosMessageName => k_RosMessageName;


        public ParallelGripperCommandActionResult() : base()
        {
            this.result = new ParallelGripperCommandResult();
        }

        public ParallelGripperCommandActionResult(HeaderMsg header, GoalStatusMsg status, ParallelGripperCommandResult result) : base(header, status)
        {
            this.result = result;
        }
        public static ParallelGripperCommandActionResult Deserialize(MessageDeserializer deserializer) => new ParallelGripperCommandActionResult(deserializer);

        ParallelGripperCommandActionResult(MessageDeserializer deserializer) : base(deserializer)
        {
            this.result = ParallelGripperCommandResult.Deserialize(deserializer);
        }
        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.header);
            serializer.Write(this.status);
            serializer.Write(this.result);
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
