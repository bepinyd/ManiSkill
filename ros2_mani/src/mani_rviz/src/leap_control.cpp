#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;

class LeapHandController : public rclcpp::Node
{
public:
  LeapHandController()
  : Node("leap_hand_cpp_controller")
  {
    // Publisher for the /joint_states topic
    publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);

    // Timer to publish at 20Hz (50ms period)
    timer_ = this->create_wall_timer(
      50ms, std::bind(&LeapHandController::timer_callback, this));

    // Initialize the 16 joint names according to your URDF
    for (int i = 0; i < 16; ++i) {
      joint_names_.push_back(std::to_string(i));
    }

    RCLCPP_INFO(this->get_logger(), "Leap Hand C++ Controller Started");
  }

private:
  void timer_callback()
  {
    auto message = sensor_msgs::msg::JointState();
    
    // Set the timestamp for RViz synchronization
    message.header.stamp = this->get_clock()->now();
    message.name = joint_names_;

    // Calculate a sine wave trajectory
    // We use the current ROS time to keep the wave smooth
    double current_time = this->get_clock()->now().seconds();
    
    // Amplitude: 0.5 radians (~30 degrees)
    // Frequency: 1.0 rad/s
    double pos_value = 0.5 * std::sin(current_time);

    // Fill the position vector with the same value for all 16 joints
    message.position.resize(16, pos_value);

    // Publish to RViz
    publisher_->publish(message);
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  std::vector<std::string> joint_names_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LeapHandController>());
  rclcpp::shutdown();
  return 0;
}