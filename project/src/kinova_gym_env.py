import gym
from gym import spaces
import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2

def compute_match_score(image, template_path='task_board_template.png'):
    """
    Computes the match score between the input image and the template.
    The score is based on the number of good keypoint matches.

    Parameters:
    - image: The current image to check for the task board.
    - template_path: Path to the template image of the task board.

    Returns:
    - match_score: A float representing the match score (0 to 1).
    - bbox: The bounding box (x_min, y_min, x_max, y_max) of the detected task board.
    """
    # Load the template and convert it to grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use ORB detector to find keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(gray_image, None)

    # Use BFMatcher to match descriptors between image and template
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate the match score (number of good matches / total number of template keypoints)
    good_matches = len(matches)
    total_keypoints = len(kp1)

    # Calculate match score as the ratio of good matches to total keypoints in the template
    match_score = good_matches / total_keypoints if total_keypoints > 0 else 0

    return match_score


class ManipulationNode:
    def __init__(self):
        rospy.init_node("ManipulationNode", anonymous=True)

        # ROS1 Subscribers
        self.joint_state_sub = rospy.Subscriber(
            "/joint_states", JointState, self.joint_state_callback)
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw", Image, self.image_callback)

        # Publisher for end-effector position
        self.ee_pose_pub = rospy.Publisher(
            "/end_effector_pose", PoseStamped, queue_size=10)

        self.joint_state = None
        self.image = None
        self.bridge = CvBridge()

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        self.joint_state = np.array(msg.position, dtype=np.float32)

    def image_callback(self, msg):
        """Callback for camera image"""
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def send_ee_command(self, position):
        """Publish end-effector position command"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "base_link"
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = position
        self.ee_pose_pub.publish(pose_msg)
        
        


class ManipulationEnv(gym.Env):
    def __init__(self):
        super(ManipulationEnv, self).__init__()

        # Initialize ROS1
        rospy.init_node("ManipulationEnv", anonymous=True)
        self.node = ManipulationNode()

        # Define observation space (joint states + camera image)
        joint_dim = 6  # Assuming a 6-DOF manipulator
        self.joint_space = spaces.Box(low=-np.pi, high=np.pi, shape=(joint_dim,), dtype=np.float32)
        self.image_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict({
            "joint_state": self.joint_space,
            "image": self.image_space
        })

        # Define action space (end-effector position: x, y, z)
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, 0.0]),  # action limits
            high=np.array([0.5, 0.5, 1.0]),
            dtype=np.float32
        )

        self.empty_joint_state = np.full(self.joint_space.shape, np.nan, dtype=np.float32)
        self.empty_image = np.zeros(self.image_space.shape, dtype=np.uint8)

    def step(self, action):
        """
        Perform an action, collect the next state, reward, and done flag.
        """
        # Send action to the robot
        self.node.send_ee_command(action)
        rospy.sleep(0.5)  # Allow time for movement

        # Wait for new data
        self.node.joint_state = None
        self.node.image = None
        observation = {"joint_state": self.empty_joint_state, "image": self.empty_image}

        while np.any(np.isnan(observation["joint_state"])) or observation["image"].sum() == 0:
            if rospy.is_shutdown():
                break
            rospy.sleep(0.1)  # Allow ROS callbacks to update data
            observation = self.get_robot_state()
            
        image_rgb = cv2.cvtColor(observation["image"], cv2.COLOR_BGR2RGB) 
        match_score = compute_match_score(image_rgb)
        # Compute reward
        reward = self.compute_reward(match_score)

        # Check if the episode is done
        done = self.check_done(match_score)

        # Additional debug info
        info = {}
        return observation, reward, done, info

    def reset(self):
        """
        Reset the environment.
        """
        self.node.send_ee_command([0.0, 0.0, 0.5])  # Reset end-effector to default position
        rospy.sleep(0.5)  # Allow time for resetting

        observation = {"joint_state": self.empty_joint_state, "image": self.empty_image}

        while np.any(np.isnan(observation["joint_state"])) or observation["image"].sum() == 0:
            if rospy.is_shutdown():
                break
            rospy.sleep(0.1)  # Allow ROS callbacks to update data
            observation = self.get_robot_state()
        
        return observation

    def get_robot_state(self):
        """
        Get the combined state of the robot (joint states + camera image).
        """
        if self.node.joint_state is None or self.node.image is None:
            return {"joint_state": self.empty_joint_state, "image": self.empty_image}
        return {"joint_state": self.node.joint_state, "image": self.node.image}

    def compute_reward(self, match_score):
        """
        Define a reward function based on the robot's state.
        """    
        # Calculate the reward
        reward = match_score  # Match score directly used as reward (scaled 0-1)
        
        # If there is no match, reward is zero
        if match_score < 0.1:
            reward = 0
        return reward

    def check_done(self, match_score):
        """
        Check if the episode is done.
        """
        
        return match_score > 0.5

    def close(self):
        """
        Clean up ROS1 resources.
        """
        rospy.signal_shutdown("Closing environment")

if __name__ == "__main__":
    env = ManipulationEnv()
    obs = env.reset()
    print("Environment initialized")
    env.close()
