using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Std;
using RosMessageTypes.Actionlib;

namespace RosMessageTypes.Control
{
    public class ParallelGripperCommandActionFeedback : ActionFeedback<ParallelGripperCommandFeedback>
    {
        public const string k_RosMessageName = "control_msgs/ParallelGripperCommandActionFeedback";
        public override string RosMessageName => k_RosMessageName;


        public ParallelGripperCommandActionFeedback() : base()
        {
            this.feedback = new ParallelGripperCommandFeedback();
        }

        public ParallelGripperCommandActionFeedback(HeaderMsg header, GoalStatusMsg status, ParallelGripperCommandFeedback feedback) : base(header, status)
        {
            this.feedback = feedback;
        }
        public static ParallelGripperCommandActionFeedback Deserialize(MessageDeserializer deserializer) => new ParallelGripperCommandActionFeedback(deserializer);

        ParallelGripperCommandActionFeedback(MessageDeserializer deserializer) : base(deserializer)
        {
            this.feedback = ParallelGripperCommandFeedback.Deserialize(deserializer);
        }
        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.header);
            serializer.Write(this.status);
            serializer.Write(this.feedback);
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
