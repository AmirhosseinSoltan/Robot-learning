# Learning-Based Robotic Manipulation for Task Versatility with Kinova Arm

## Team Members
* achaur2s
* asoltas
* gchen2s
* nselva2s

## Problem statement 

Robotics can augment or replace repetitive and intricate human actions. The ability to perform versatile, unstructured manipulation tasks is essential for robots. This enables them to assist with a variety of manual tasks with diverse settings, enhancing functionality and autonomy.

Traditional programming and predefined motion paths would lack the flexibility required for consistent and accurate task execution under changing conditions. Reinforcement learning (RL) presents a suitable solution by allowing the robot to autonomously learn effective policies for each task. With RL, the robot can develop robust strategies to generalize across tasks with complex input-output mappings, adapting to dynamic variations in object placements and control requirements. RL also enables the development of reusable skills, making it possible to generalize learned policies for other similar manipulation tasks.

In our project, we aim to develop a learning-based robotic manipulation framework for versatile manipulation tasks using the 7-dof Kinova Gen3 [(link)](https://www.kinovarobotics.com/product/gen3-robots) with a Robotiq 2F-85 gripper [(link)](https://robotiq.com/products/2f85-140-adaptive-robot-gripper) on the task board.

<!-- <p float="left">
  <img src="https://www.kinovarobotics.com/uploads/_2000xAUTO_crop_center-center_none/22037/Gen3-robot-img-Cover-img-is-loaded-block-1B.webp" width=23% height="290"/>
  <img src="https://github.com/b-it-bots/robothon2023/raw/main/docs/images/platform/gripper_with_foam.jpg" width="280" />
  <img src="https://user-images.githubusercontent.com/47410011/230386146-407067bd-04dd-4105-892f-4292a32af506.jpg" width="250" height="290"/>
</p> -->
<p float="left">
  <figure style="display:inline-block; text-align: center; margin-right: 20px;">
    <img src="./images/kinova.png" width="100%" height="290"/>
    <figcaption>Fig.1 Kinova Gen3 Robot</figcaption>
  </figure>
  <figure style="display:inline-block; text-align: center; margin-right: 20px;">
    <img src="./images/robotiq.png" width="280" />
    <figcaption>Fig.2 Robotiq Gripper with Foam</figcaption>
  </figure>
  <figure style="display:inline-block; text-align: center;">
    <img src="./images/task_board.png" width="250" height="290"/>
    <figcaption>Fig.3 Task board</figcaption>
  </figure>
</p>


We intend to address the following tasks inspired by the Robothon Challenge in the euROBIN MVSC Competition:

**Minimum viable : Localization and pressing start trial button (Task 1)**: 

As the task board is placed at a random orientation on the table, to perform each task, first the task board needs to be localized. Based on the detected board origin, the arm is moved to a position just above the blue button (i.e. the start trial button), press the button and move back up.

**Expected: Impedance control/sliding the multimeter (Task 2)**: 
1. The arm is moved above the initial location of the slider, and moved down until contact with the board 
2. The robot will learn to adjust the slider to match the target marker on the screen, then moves back to its original position.

**Desired: Plug probe in test port(Task 3)**: 

***Grasp plug***

1. To successfully grasp the plug from the board, the arm first moves above the plug and align itself with the plug. 
2. Once it is aligned, the arm moves down until it makes contact with the board. The gripper is then closed to firmly grasp the plug. 
3. Finally, the arm moves back up, carefully holding the plug. 


***Insert plug in test port***

After alignment, the plug is inserted into the test port, by moving downwards. Then, it moves back to its original position.



## Learning Techniques

In our project, we will be using Adaptive Curriculum Generation from Demonstrations (ACGD) from [1]. This approach best suits our problem statement of localization w.r.t to the task board and performing tasks by using the architecture of the policy network as mentioned in Fig.4.

<p float="left">
  <figure style="display:center; text-align: center;">
    <img src="./images/policy_arch.png" />
    <figcaption>Fig.4 Architecture of the policy network [1]</figcaption>
  </figure>
</p>

## Learning Data

We will be using different data for each task
- Task 1: We will collect a trajectory data of the Kinova arm using joystick and record both camera feed and end-effector pose data. Using this data we will train the model as mentioned in the paper [1].
- Task 2: For this task, initially we will need to record data from position of task 1 to start of task 2 using same process as used in task 1. Then we will need another set of data to record the various positions of the pointers in the display. 
- Task 3: Data from position of task 2 to start of task 3 will be collected for demonstration purpose and another policy will be used to pick up the probe plug from the starting port and insert it into the testing port [2].

## Expected Project Outcomes

List and briefly describe the main expected outcomes of your project (e.g. a simulation environment or a trained policy for your problem of interest).
1. #### Task Board Localization and Button Pressing Module:
    A trained policy capable of accurately localizing the task board in varying orientations on the table and autonomously pressing the start trial button. This module will demonstrate robustness to positional variations and initial conditions.
2. #### Slider Adjustment Policy:
    A policy trained to execute impedance control for precise sliding of the multimeter slider to the desired position. The outcome will include smooth motion control, adaptability to marker variations on the display, and effective feedback integration.
3. #### Plug Grasping and Insertion Framework:
    A complete policy framework for grasping a plug from the task board and inserting it into the test port. This includes a trained policy for alignment, grasping, and insertion tasks, demonstrating dexterous manipulation and collision-free operation.
4. #### Comprehensive Dataset for RL Training:
    A structured dataset containing trajectories, end-effector poses, and vision data for all three tasks. This dataset will be reusable for further research and extension of robotic manipulation tasks.
5. #### Visualization and Reporting on IoT Dashboard:
    Integration of the evaluation results and task performance metrics into an IoT web dashboard for real-time monitoring and reporting. Metrics such as execution time, success rate, and robustness will be visualized to track progress.

## Evaluation Plan

1. #### Accuracy of Task Completion:
    - Localization and pressing the start trial button.
    - Sliding the multimeter slider to match the target marker.
    - Grasping and inserting the plug into the test port.
2. #### Task Board Automated Solution Factors:
    Execution Speed, Robustness, Documentation. Demonstrations on the task board will be monitored, aggregated and reported over this [IoT web dashboard ](https://cloud.kaaiot.com/wd-public/c1v9jqmgul2l1s47m6bg/dashboards/0d2f0e4c-6a80-4cf4-a48d-5b25fcb35ac0/c3f502ea-b923-492f-b3c3-1a894e67afd6?public_id=4e4990d1-dcab-4f1a-b1a6-8648e87bc5ad).


## References

1. L. Hermann, M. Argus, A. Eitel, A. Amiranashvili, W. Burgard and T. Brox, "Adaptive Curriculum Generation from Demonstrations for Sim-to-Real Visuomotor Control," 2020 IEEE International Conference on Robotics and Automation (ICRA), Paris, France, 2020, pp. 6498-6505, doi: 10.1109/ICRA40945.2020.9197108. [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9197108)
2. euRobin-MVSC Challenge [Link](https://sites.google.com/view/eurobin-msvc/)
3. euRobin-MVSC Challenge IoT Web Dashboard [Link](https://cloud.kaaiot.com/wd-public/c1v9jqmgul2l1s47m6bg/dashboards/0d2f0e4c-6a80-4cf4-a48d-5b25fcb35ac0/c3f502ea-b923-492f-b3c3-1a894e67afd6?public_id=4e4990d1-dcab-4f1a-b1a6-8648e87bc5ad)
4. Kinova gen3 arm [Link](https://www.kinovarobotics.com/product/gen3-robots)

