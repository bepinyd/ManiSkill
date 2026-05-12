import sapien

# 1. Create a basic SAPIEN engine and scene (no ManiSkill overhead needed)
scene = sapien.Scene()

# 2. Load the URDF using SAPIEN's built-in loader
loader = scene.create_urdf_loader()
# This creates a 'template' of the robot in the scene
robot_builder = loader.load("/home/bipin/ManiSkill/leap_hand/leap_hand_left.urdf")

# 3. Print all joint names
print("--- All Joints ---")
for joint in robot_builder.get_joints():
    print(joint.name)

# 4. Print only active (motorized) joint names
print("\n--- Active Joints (Motorized) ---")
active_joint_names = [j.name for j in robot_builder.get_active_joints()]
print(active_joint_names)