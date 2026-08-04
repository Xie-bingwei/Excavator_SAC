using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;


namespace RosMessageTypes.Control
{
    public class ParallelGripperCommandAction : Action<ParallelGripperCommandActionGoal, ParallelGripperCommandActionResult, ParallelGripperCommandActionFeedback, ParallelGripperCommandGoal, ParallelGripperCommandResult, ParallelGripperCommandFeedback>
    {
        public const string k_RosMessageName = "control_msgs/ParallelGripperCommandAction";
        public override string RosMessageName => k_RosMessageName;


        public ParallelGripperCommandAction() : base()
        {
            this.action_goal = new ParallelGripperCommandActionGoal();
            this.action_result = new ParallelGripperCommandActionResult();
            this.action_feedback = new ParallelGripperCommandActionFeedback();
        }

        public static ParallelGripperCommandAction Deserialize(MessageDeserializer deserializer) => new ParallelGripperCommandAction(deserializer);

        ParallelGripperCommandAction(MessageDeserializer deserializer)
        {
            this.action_goal = ParallelGripperCommandActionGoal.Deserialize(deserializer);
            this.action_result = ParallelGripperCommandActionResult.Deserialize(deserializer);
            this.action_feedback = ParallelGripperCommandActionFeedback.Deserialize(deserializer);
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.action_goal);
            serializer.Write(this.action_result);
            serializer.Write(this.action_feedback);
        }

    }
}
