# Use the official ROS Noetic image as a base
FROM ros:noetic-ros-base

# Set the working directory inside the container
WORKDIR /root/catkin_ws

# Install dependencies (Python, git, ROS packages, etc.)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ros-noetic-diagnostic-updater \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies
# Install Python dependencies (only once for transitions version 0.7.2)
RUN pip3 install --force-reinstall transitions==0.7.2
RUN pip3 install pyserial

# Create the catkin workspace
RUN mkdir -p /root/catkin_ws/src

# Copy the local repository into the container
COPY ./src/weiss_gripper_ieg76 /root/catkin_ws/src/weiss_gripper_ieg76

# Ensure Python 3 is set as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd /root/catkin_ws && catkin_make"

# Source ROS environment and workspace when the container starts
CMD ["bash", "-c", "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && exec roslaunch weiss_gripper_ieg76 srv_client.launch"]



